"""Turn a general photo into a 250x250 LFW-style image around its largest face."""

import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ml.preprocess import DEFAULT_CONFIG, LFW_IMAGE_SIZE, preprocess

YUNET_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
MAX_DETECTION_SIDE = 640  # larger photos are downscaled for detection only
MIN_DETECTION_SIDE = 32  # YuNet can return NaN boxes on images thinner than this


@dataclass(frozen=True)
class FaceCropConfig:
    """Expansion and shift minimize pixel error against LFW framing; see docs/FACE_DETECTION.md."""

    detector: str = "yunet-2023mar"
    score_threshold: float = 0.6
    expansion: float = 1.95  # square side, as a multiple of the detected box's longer side
    y_shift: float = -0.02  # moves the square up, as a fraction of its side

    def to_manifest(self) -> dict:
        return asdict(self)


DEFAULT_FACE_CROP = FaceCropConfig()


class FaceCropper:
    def __init__(self, config: FaceCropConfig = DEFAULT_FACE_CROP, model_path: Path = YUNET_MODEL):
        self.config = config
        self._detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), config.score_threshold
        )
        self._lock = threading.Lock()  # the detector holds its input size as mutable state

    def _largest_box(self, image: Image.Image) -> np.ndarray | None:
        scale = min(1.0, MAX_DETECTION_SIDE / max(image.size))
        small = image
        if scale < 1.0:
            small = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            )
        if min(small.size) < MIN_DETECTION_SIDE:
            return None
        bgr = cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2BGR)
        with self._lock:
            self._detector.setInputSize(small.size)
            _, faces = self._detector.detect(bgr)
        if faces is None:
            return None
        boxes = faces[:, :4]
        boxes = boxes[np.isfinite(boxes).all(axis=1) & (boxes[:, 2] > 0) & (boxes[:, 3] > 0)]
        if len(boxes) == 0:
            return None
        return max(boxes, key=lambda box: box[2] * box[3]) / scale

    def __call__(self, image: Image.Image) -> Image.Image | None:
        """Return the cropped face, or None if no face was found."""
        rgb = image.convert("RGB")
        box = self._largest_box(rgb)
        if box is None:
            return None
        x, y, width, height = box
        side = max(width, height) * self.config.expansion
        center_x = x + width / 2
        center_y = y + height / 2 + self.config.y_shift * side
        region = tuple(
            int(round(value))
            for value in (
                center_x - side / 2,
                center_y - side / 2,
                center_x + side / 2,
                center_y + side / 2,
            )
        )
        return rgb.crop(region).resize(LFW_IMAGE_SIZE, Image.Resampling.BICUBIC)


@dataclass(frozen=True)
class Faces:
    images: np.ndarray  # preprocessed faces, detected ones only
    detected: np.ndarray  # boolean mask over the input paths


def crop_faces(paths, cropper: FaceCropper) -> Faces:
    faces = []
    detected = []
    for path in paths:
        with Image.open(path) as image:
            cropped = cropper(image)
        detected.append(cropped is not None)
        if cropped is not None:
            faces.append(preprocess(cropped, DEFAULT_CONFIG))
    return Faces(np.stack(faces), np.array(detected))
