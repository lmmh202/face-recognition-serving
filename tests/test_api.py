import io
import logging
import math

import numpy as np
import pytest
from PIL import Image

from app.logging import logger
from app.main import MAX_BODY_BYTES, MAX_PIXELS
from ml.train import load_dataset

MODELS = ["cnn", "pca-knn"]


def gradient_image(size: tuple[int, int] = (250, 250)) -> Image.Image:
    width, height = size
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(255, 0, height, dtype=np.uint8)
    red, green = np.meshgrid(x, y)
    pixels = np.dstack([red, green, np.full_like(red, 128)])
    return Image.fromarray(pixels)


def encode(image: Image.Image, image_format: str = "JPEG", **options) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format, **options)
    return buffer.getvalue()


def post(client, data: bytes, params: dict | None = None, filename: str = "face.jpg"):
    return client.post("/predict", params=params, files={"file": (filename, data)})


@pytest.mark.parametrize("model", MODELS)
def test_predict_response_schema(client, model):
    response = post(client, encode(gradient_image()), {"model": model})
    assert response.status_code == 200

    body = response.json()
    assert set(body) == {"model", "model_version", "predictions", "inference_ms", "request_id"}
    assert body["model"] == model
    assert body["model_version"] == "v1"
    assert body["inference_ms"] >= 0
    assert body["request_id"] == response.headers["X-Request-ID"]

    predictions = body["predictions"]
    assert len(predictions) == 3
    assert len({p["name"] for p in predictions}) == 3
    probabilities = [p["probability"] for p in predictions]
    assert probabilities == sorted(probabilities, reverse=True)
    assert all(0 <= p <= 1 for p in probabilities)
    assert sum(probabilities) <= 1 + 1e-6


def test_default_model_is_cnn(client):
    assert post(client, encode(gradient_image())).json()["model"] == "cnn"


def test_png_is_accepted(client):
    assert post(client, encode(gradient_image(), "PNG")).status_code == 200


def test_exif_rotation_is_corrected(client):
    original = gradient_image()
    exif = Image.Exif()
    exif[0x0112] = 6  # display needs a 270 degree counter-clockwise rotation
    rotated = encode(original.rotate(90), exif=exif, quality=100)

    upright = post(client, encode(original, quality=100)).json()["predictions"][0]
    corrected = post(client, rotated).json()["predictions"][0]
    assert corrected["name"] == upright["name"]
    assert corrected["probability"] == pytest.approx(upright["probability"], abs=0.02)


def test_undecodable_file_is_400(client):
    assert post(client, b"not an image").status_code == 400


def test_truncated_image_is_400(client):
    data = encode(gradient_image())
    assert post(client, data[: len(data) // 2]).status_code == 400


def test_oversized_upload_is_413(client):
    assert post(client, b"\0" * (MAX_BODY_BYTES + 1)).status_code == 413


def test_too_many_pixels_is_413(client):
    side = math.isqrt(MAX_PIXELS) + 1
    data = encode(Image.new("L", (side, side)), "PNG")
    assert len(data) < MAX_BODY_BYTES
    assert post(client, data).status_code == 413


def test_unsupported_format_is_415(client):
    assert post(client, encode(gradient_image(), "GIF")).status_code == 415


def test_wrong_image_size_is_422(client):
    assert post(client, encode(gradient_image((100, 100)))).status_code == 422


def test_invalid_model_is_422(client):
    assert post(client, encode(gradient_image()), {"model": "nope"}).status_code == 422


def test_missing_file_is_422(client):
    assert client.post("/predict").status_code == 422


def test_health_reports_loaded_models(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "loaded_models": MODELS}


def test_models_summary(client):
    body = client.get("/models").json()
    assert [entry["model"] for entry in body] == MODELS
    for entry in body:
        assert entry["model_version"] == "v1"
        assert entry["num_labels"] == 62
        assert entry["input_size"] == {"height": 87, "width": 65}


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_request_log_has_fields_and_no_upload_details(client):
    handler = _ListHandler()
    logger.addHandler(handler)
    try:
        response = post(client, encode(gradient_image()), filename="private_person_name.jpg")
    finally:
        logger.removeHandler(handler)

    fields = handler.records[-1].fields
    assert fields["request_id"] == response.json()["request_id"]
    assert fields["status_code"] == 200
    assert fields["model"] == "cnn"
    assert fields["model_version"] == "v1"
    assert fields["preprocess_ms"] >= 0
    assert fields["inference_ms"] >= 0
    assert "private_person_name" not in repr(fields)


@pytest.mark.parametrize("model", MODELS)
def test_api_accuracy_matches_manifest(client, require_lfw, model):
    """The whole upload path must reproduce the accuracy measured at training time."""
    data = load_dataset()
    correct = 0
    for path, label in zip(data.test_paths, data.y_test, strict=True):
        with open(path, "rb") as image:
            response = post(client, image.read(), {"model": model})
        assert response.status_code == 200
        correct += response.json()["predictions"][0]["name"] == data.labels[label]

    recorded = next(m for m in client.get("/models").json() if m["model"] == model)
    assert correct / len(data.y_test) == pytest.approx(recorded["test_accuracy"])
