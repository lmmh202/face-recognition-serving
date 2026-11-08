"""Image preprocessing shared by training and serving.

Reproduces sklearn's fetch_lfw_people (sklearn/datasets/_lfw.py::_load_imgs):
crop the RGB image, resize it with Pillow's default BICUBIC filter, scale to
[0, 1], then average the color channels. tests/test_preprocess.py checks this
against the arrays sklearn produces.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image

LFW_IMAGE_SIZE = (250, 250)
GRAYSCALE_METHOD = "channel_mean"

_RESAMPLING = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
}


@dataclass(frozen=True)
class PreprocessConfig:
    y_range: tuple[int, int] = (70, 195)
    x_range: tuple[int, int] = (78, 172)
    resize: float = 0.7
    interpolation: str = "bicubic"

    @property
    def slices(self) -> tuple[slice, slice]:
        return slice(*self.y_range), slice(*self.x_range)

    @property
    def output_size(self) -> tuple[int, int]:
        """(height, width), truncated the same way as sklearn."""
        height = self.y_range[1] - self.y_range[0]
        width = self.x_range[1] - self.x_range[0]
        return int(self.resize * height), int(self.resize * width)

    def to_manifest(self) -> dict:
        return {
            "color": False,
            "grayscale": GRAYSCALE_METHOD,
            "slice": [list(self.y_range), list(self.x_range)],
            "resize": self.resize,
            "interpolation": self.interpolation,
        }

    @classmethod
    def from_manifest(cls, preprocessing: dict) -> "PreprocessConfig":
        if preprocessing["color"] or preprocessing["grayscale"] != GRAYSCALE_METHOD:
            raise ValueError(f"unsupported preprocessing: {preprocessing}")
        y_range, x_range = preprocessing["slice"]
        return cls(
            y_range=tuple(y_range),
            x_range=tuple(x_range),
            resize=preprocessing["resize"],
            interpolation=preprocessing["interpolation"],
        )


DEFAULT_CONFIG = PreprocessConfig()


def preprocess(image: Image.Image, config: PreprocessConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Return a float32 (height, width) array in [0, 1] from a 250x250 LFW-style image."""
    if image.size != LFW_IMAGE_SIZE:
        raise ValueError(f"expected a {LFW_IMAGE_SIZE} image, got {image.size}")

    height, width = config.output_size
    top, bottom = config.y_range
    left, right = config.x_range
    resized = (
        image.convert("RGB")
        .crop((left, top, right, bottom))
        .resize((width, height), _RESAMPLING[config.interpolation])
    )
    return (np.asarray(resized, dtype=np.float32) / 255.0).mean(axis=2)
