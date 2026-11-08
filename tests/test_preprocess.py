import numpy as np
import pytest
from PIL import Image

from ml.preprocess import DEFAULT_CONFIG, PreprocessConfig, preprocess
from ml.train import fetch_lfw, lfw_image_paths


def test_golden_matches_fetch_lfw_people():
    try:
        expected = fetch_lfw().images
        paths = lfw_image_paths()
    except (OSError, FileNotFoundError) as error:
        pytest.skip(f"LFW dataset unavailable: {error}")

    assert len(paths) == len(expected)
    worst = 0.0
    for path, target in zip(paths, expected, strict=True):
        with Image.open(path) as image:
            worst = max(worst, float(np.abs(preprocess(image) - target).max()))
    assert worst <= 1e-5, f"largest pixel difference: {worst}"


def test_output_shape_and_range():
    image = Image.fromarray(np.full((250, 250, 3), 255, dtype=np.uint8))
    face = preprocess(image)
    assert face.shape == (87, 65)
    assert face.dtype == np.float32
    assert face.min() == pytest.approx(1.0) and face.max() == pytest.approx(1.0)


def test_grayscale_is_unweighted_channel_mean():
    pixels = np.zeros((250, 250, 3), dtype=np.uint8)
    pixels[..., 1] = 255  # pure green: mean is 1/3, luma would be about 0.587
    assert preprocess(Image.fromarray(pixels))[0, 0] == pytest.approx(1 / 3)


def test_rejects_non_lfw_size():
    with pytest.raises(ValueError):
        preprocess(Image.new("RGB", (100, 100)))


def test_config_round_trips_through_manifest():
    assert PreprocessConfig.from_manifest(DEFAULT_CONFIG.to_manifest()) == DEFAULT_CONFIG
