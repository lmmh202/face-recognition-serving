"""Load saved models and expose one prediction interface for both."""

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

from app.schemas import ModelName
from ml.face_crop import FaceCropConfig, FaceCropper
from ml.preprocess import PreprocessConfig

DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


@dataclass(frozen=True)
class LoadedModel:
    manifest: dict
    config: PreprocessConfig
    face_cropper: FaceCropper
    predict_proba: Callable[[np.ndarray], np.ndarray]  # (h, w) face -> class probabilities

    @property
    def version(self) -> str:
        return self.manifest["model_version"]

    @property
    def unknown_threshold(self) -> float:
        return self.manifest["unknown_threshold"]

    @property
    def labels(self) -> list[str]:
        return self.manifest["labels"]


def _load(directory: Path, predict_proba: Callable[[np.ndarray], np.ndarray]) -> LoadedModel:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    return LoadedModel(
        manifest=manifest,
        config=PreprocessConfig.from_manifest(manifest["preprocessing"]),
        face_cropper=FaceCropper(FaceCropConfig(**manifest["face_crop"])),
        predict_proba=predict_proba,
    )


def _load_cnn(directory: Path) -> LoadedModel:
    model = tf.keras.models.load_model(directory / "model.keras")

    def predict_proba(face: np.ndarray) -> np.ndarray:
        batch = face[np.newaxis, ..., np.newaxis]
        return model(batch, training=False).numpy()[0]

    return _load(directory, predict_proba)


def _load_pca_knn(directory: Path) -> LoadedModel:
    pipeline = joblib.load(directory / "model.joblib")

    def predict_proba(face: np.ndarray) -> np.ndarray:
        return pipeline.predict_proba(face.reshape(1, -1))[0]

    return _load(directory, predict_proba)


def load_models() -> dict[ModelName, LoadedModel]:
    """Raises if an artifact or calibration is missing, so the service fails at startup."""
    root = Path(os.environ.get("ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR))
    return {
        ModelName.cnn: _load_cnn(root / "cnn-v1"),
        ModelName.pca_knn: _load_pca_knn(root / "pca-knn-v1"),
    }
