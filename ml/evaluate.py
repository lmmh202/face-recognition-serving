"""Reload saved models and check test accuracy against their manifests.

Run with: python -m ml.evaluate  (exits non-zero on a mismatch)
"""

import json
import sys
from pathlib import Path

import joblib
import tensorflow as tf

from ml.train import (
    CNN_DIR,
    CNN_FILE,
    PCA_KNN_DIR,
    PCA_KNN_FILE,
    cnn_accuracy,
    load_dataset,
    pca_knn_accuracy,
)
from ml.train_cnn_v2 import CNN_V2_DIR


def recorded_accuracy(directory: Path) -> float:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    return manifest["test_accuracy"]


def main() -> int:
    data = load_dataset()
    results = {
        "cnn": (
            cnn_accuracy(tf.keras.models.load_model(CNN_DIR / CNN_FILE), data),
            recorded_accuracy(CNN_DIR),
        ),
        "cnn-v2": (
            cnn_accuracy(tf.keras.models.load_model(CNN_V2_DIR / CNN_FILE), data),
            recorded_accuracy(CNN_V2_DIR),
        ),
        "pca-knn": (
            pca_knn_accuracy(joblib.load(PCA_KNN_DIR / PCA_KNN_FILE), data),
            recorded_accuracy(PCA_KNN_DIR),
        ),
    }
    ok = True
    for name, (reloaded, recorded) in results.items():
        match = reloaded == recorded
        ok = ok and match
        print(f"{name}: reloaded={reloaded:.4f} recorded={recorded:.4f} match={match}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
