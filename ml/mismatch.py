"""Measure how much test accuracy drops when serving preprocessing differs.

Models are the saved artifacts; only the test images are preprocessed differently.
Run with: python -m ml.mismatch
"""

from collections.abc import Callable
from dataclasses import replace

import joblib
import numpy as np
import tensorflow as tf
from PIL import Image

from ml.preprocess import DEFAULT_CONFIG, preprocess
from ml.train import (
    CNN_DIR,
    CNN_FILE,
    PCA_KNN_DIR,
    PCA_KNN_FILE,
    Dataset,
    cnn_accuracy,
    load_dataset,
    pca_knn_accuracy,
)


def luma_grayscale(image: Image.Image) -> np.ndarray:
    height, width = DEFAULT_CONFIG.output_size
    top, bottom = DEFAULT_CONFIG.y_range
    left, right = DEFAULT_CONFIG.x_range
    gray = image.crop((left, top, right, bottom)).resize((width, height)).convert("L")
    return np.asarray(gray, dtype=np.float32) / 255.0


def no_crop(image: Image.Image) -> np.ndarray:
    height, width = DEFAULT_CONFIG.output_size
    return (np.asarray(image.resize((width, height)), dtype=np.float32) / 255.0).mean(axis=2)


def shifted_crop(image: Image.Image) -> np.ndarray:
    config = replace(DEFAULT_CONFIG, y_range=(80, 205), x_range=(88, 182))
    return preprocess(image, config)


VARIANTS: dict[str, Callable[[Image.Image], np.ndarray]] = {
    "shared preprocess (baseline)": preprocess,
    "PIL convert('L') grayscale": luma_grayscale,
    "bilinear resize": lambda i: preprocess(i, replace(DEFAULT_CONFIG, interpolation="bilinear")),
    "nearest resize": lambda i: preprocess(i, replace(DEFAULT_CONFIG, interpolation="nearest")),
    "crop shifted by 10px": shifted_crop,
    "no crop (whole image resized)": no_crop,
    "no [0, 1] scaling": lambda i: preprocess(i) * 255.0,
}


def with_test_images(data: Dataset, transform: Callable[[Image.Image], np.ndarray]) -> Dataset:
    faces = []
    for path in data.test_paths:
        with Image.open(path) as image:
            faces.append(transform(image))
    images = np.stack(faces)
    return replace(data, img_test=images, x_test=images.reshape(len(images), -1))


def main() -> None:
    data = load_dataset()
    cnn = tf.keras.models.load_model(CNN_DIR / CNN_FILE)
    pca_knn = joblib.load(PCA_KNN_DIR / PCA_KNN_FILE)

    print("| Serving preprocessing | CNN | PCA+KNN |")
    print("|---|---|---|")
    for name, transform in VARIANTS.items():
        variant = with_test_images(data, transform)
        print(
            f"| {name} | {cnn_accuracy(cnn, variant) * 100:.2f}% "
            f"| {pca_knn_accuracy(pca_knn, variant) * 100:.2f}% |"
        )


if __name__ == "__main__":
    main()
