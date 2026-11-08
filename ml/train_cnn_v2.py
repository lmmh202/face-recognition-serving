"""Train cnn-v2: every non-test LFW image, framed the way the service frames photos.

Differences from v1: no 20-per-person cap (with class weights), augmentation, and
training on face-detection crops. The test set is the same 310 images as v1.
Run with: python -m ml.train_cnn_v2, then python -m ml.calibrate
"""

from dataclasses import dataclass

import keras
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from ml.face_crop import FaceCropper, crop_faces
from ml.train import (
    ARTIFACTS_DIR,
    CNN_BATCH_SIZE,
    CNN_FILE,
    CNN_VAL_SIZE,
    RANDOM_SEED,
    Dataset,
    build_manifest,
    cnn_accuracy,
    create_cnn,
    fetch_lfw,
    lfw_image_paths,
    load_dataset,
    write_manifest,
)

MODEL_NAME = "cnn-v2"
MODEL_VERSION = "v2"
CNN_V2_DIR = ARTIFACTS_DIR / MODEL_NAME
MAX_EPOCHS = 120
EARLY_STOPPING_PATIENCE = 15


@dataclass(frozen=True)
class Pool:
    images: np.ndarray  # detection crops; the funneled crop where no face was found
    y: np.ndarray
    detected: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray


def load_pool(data: Dataset, cropper: FaceCropper) -> Pool:
    lfw = fetch_lfw()
    paths = lfw_image_paths()
    keep = ~np.isin(paths, data.test_paths)
    images = lfw.images[keep]
    y = lfw.target[keep]

    faces = crop_faces(paths[keep], cropper)
    images[faces.detected] = faces.images

    train_idx, val_idx = train_test_split(
        np.arange(len(y)), test_size=CNN_VAL_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    return Pool(images, y, faces.detected, train_idx, val_idx)


def build_model(input_shape: tuple[int, int, int], num_classes: int) -> keras.Model:
    """The v1 architecture behind augmentation layers, which are inactive at inference."""
    base = create_cnn(input_shape, num_classes)
    inputs = keras.Input(input_shape)
    x = keras.layers.RandomFlip("horizontal", seed=0)(inputs)
    x = keras.layers.RandomTranslation(0.06, 0.06, fill_mode="nearest", seed=1)(x)
    x = keras.layers.RandomRotation(0.03, fill_mode="nearest", seed=2)(x)
    x = keras.layers.RandomZoom(0.08, fill_mode="nearest", seed=3)(x)
    model = keras.Model(inputs, base(x))
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    tf.config.experimental.enable_op_determinism()
    data = load_dataset()
    pool = load_pool(data, FaceCropper())
    train_images = pool.images[pool.train_idx][..., np.newaxis]
    val_images = pool.images[pool.val_idx][..., np.newaxis]
    train_y, val_y = pool.y[pool.train_idx], pool.y[pool.val_idx]
    classes = np.arange(len(data.labels))
    class_weight = dict(
        enumerate(compute_class_weight("balanced", classes=classes, y=train_y))
    )

    tf.keras.utils.set_random_seed(RANDOM_SEED)
    model = build_model((data.height, data.width, 1), len(data.labels))
    history = model.fit(
        train_images,
        train_y,
        validation_data=(val_images, val_y),
        epochs=MAX_EPOCHS,
        batch_size=CNN_BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )

    accuracy = cnn_accuracy(model, data)
    CNN_V2_DIR.mkdir(parents=True, exist_ok=True)
    model.save(CNN_V2_DIR / CNN_FILE)
    write_manifest(
        CNN_V2_DIR,
        build_manifest(
            MODEL_NAME,
            accuracy,
            data,
            model_version=MODEL_VERSION,
            training={
                "images": len(pool.y),
                "framing": "face_detection",
                "class_weight": "balanced",
                "augmentation": True,
                "epochs_run": len(history.history["loss"]),
            },
        ),
    )
    print(f"{MODEL_NAME} test (no detection)={accuracy:.4f}")


if __name__ == "__main__":
    main()
