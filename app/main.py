import io
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from PIL import Image, ImageOps
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.logging import configure_logging, logger
from app.registry import load_models
from app.schemas import HealthResponse, ModelName, ModelSummary, PredictResponse
from ml.preprocess import preprocess

MAX_BODY_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 20_000_000
ALLOWED_FORMATS = {"JPEG", "PNG"}
TOP_K = 3
UNKNOWN_IDENTITY = "unknown"


class PredictBodyLimit:
    """Reject oversized /predict bodies while reading them, before multipart parsing."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] != "/predict":
            await self.app(scope, receive, send)
            return

        messages: list[Message] = []
        total = 0
        while True:
            message = await receive()
            messages.append(message)
            total += len(message.get("body", b""))
            if total > MAX_BODY_BYTES:
                response = JSONResponse({"detail": "upload too large"}, status_code=413)
                await response(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        replay = iter(messages)

        async def replay_receive() -> Message:
            return next(replay, None) or await receive()

        await self.app(scope, replay_receive, send)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    app.state.models = load_models()
    logger.info("models_loaded", extra={"fields": {"models": list(app.state.models)}})
    yield


DESCRIPTION = """
LFW 데이터셋의 공인 62명 중 업로드한 얼굴과 가장 닮은 사람을 고르는 데모입니다.

- 신원 확인 용도로 쓰면 안 됩니다. 학습에 없는 사람의 사진에도 62명 중 한 명의 이름을 돌려줄 수 있습니다.
- 업로드한 이미지는 저장하거나 로그에 남기지 않습니다.
"""

app = FastAPI(title="face-recognition-serving", description=DESCRIPTION, lifespan=lifespan)
app.add_middleware(PredictBodyLimit)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    request.state.log_fields = {}
    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        fields = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": status_code,
            "total_ms": round((time.perf_counter() - started) * 1000, 2),
            **request.state.log_fields,
        }
        logger.info("request", extra={"fields": fields})


def decode_image(data: bytes) -> Image.Image:
    """Validate format and pixel count from the header, then decode."""
    try:
        image = Image.open(io.BytesIO(data))
    except Image.DecompressionBombError as error:
        raise HTTPException(413, "image has too many pixels") from error
    except Exception as error:  # Pillow plugins raise many types on malformed input
        raise HTTPException(400, "could not decode image") from error

    if image.format not in ALLOWED_FORMATS:
        raise HTTPException(415, f"unsupported image format: {image.format}")
    width, height = image.size
    if width * height > MAX_PIXELS:
        raise HTTPException(413, "image has too many pixels")

    try:
        image.load()
        return ImageOps.exif_transpose(image)
    except Exception as error:
        raise HTTPException(400, "could not decode image") from error


def decide_identity(name: str, probability: float, threshold: float) -> str:
    return name if probability >= threshold else UNKNOWN_IDENTITY


# A plain def: FastAPI runs it in a thread pool, so CPU inference does not block the event loop.
@app.post("/predict", response_model=PredictResponse)
def predict(
    request: Request,
    file: UploadFile = File(),
    model: ModelName = Query(default=ModelName.cnn_v2),
    detect: bool = Query(default=True, description="false: input is already LFW-framed 250x250"),
) -> PredictResponse:
    loaded = request.app.state.models[model]
    fields = request.state.log_fields
    fields.update(model=model.value, model_version=loaded.version, detect=detect)

    image = decode_image(file.file.read())
    if detect:
        detect_started = time.perf_counter()
        image = loaded.face_cropper(image)
        fields["detection_ms"] = round((time.perf_counter() - detect_started) * 1000, 2)
        if image is None:
            raise HTTPException(422, "no face detected")

    started = time.perf_counter()
    try:
        face = preprocess(image, loaded.config)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    preprocessed = time.perf_counter()
    probabilities = loaded.predict_proba(face)
    finished = time.perf_counter()

    preprocess_ms = round((preprocessed - started) * 1000, 2)
    inference_ms = round((finished - preprocessed) * 1000, 2)
    fields.update(preprocess_ms=preprocess_ms, inference_ms=inference_ms)

    top = np.argsort(-probabilities, kind="stable")[:TOP_K]
    return PredictResponse(
        model=model,
        model_version=loaded.version,
        identity=decide_identity(
            loaded.labels[top[0]], float(probabilities[top[0]]), loaded.unknown_threshold
        ),
        predictions=[
            {"name": loaded.labels[i], "probability": float(probabilities[i])} for i in top
        ],
        inference_ms=inference_ms,
        request_id=request.state.request_id,
    )


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    return HealthResponse(status="ok", loaded_models=list(request.app.state.models))


@app.get("/models", response_model=list[ModelSummary])
def models(request: Request) -> list[ModelSummary]:
    return [
        ModelSummary(
            model=name,
            model_version=loaded.version,
            num_labels=len(loaded.labels),
            input_size=loaded.manifest["input_size"],
            test_accuracy=loaded.manifest["test_accuracy"],
            unknown_threshold=loaded.unknown_threshold,
        )
        for name, loaded in request.app.state.models.items()
    ]
