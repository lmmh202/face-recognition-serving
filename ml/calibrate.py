"""Set each model's "unknown" threshold and record it in its manifest.

Scores come through the face detection path, as in serving. People outside the
62 classes are split into calibration and evaluation halves; the test set is
only used for the final report. Run after ml.train:

    python -m ml.calibrate
"""

import json

import joblib
import numpy as np
import tensorflow as tf
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split

from ml.face_crop import DEFAULT_FACE_CROP, FaceCropper, crop_faces
from ml.train import (
    CNN_DIR,
    CNN_FILE,
    CNN_VAL_SIZE,
    CV_FOLDS,
    MIN_FACES_PER_PERSON,
    PCA_KNN_DIR,
    PCA_KNN_FILE,
    RANDOM_SEED,
    lfw_person_dirs,
    load_dataset,
    write_manifest,
)
from ml.train_cnn_v2 import CNN_V2_DIR, load_pool

TARGET_UNKNOWN_ACCEPT_RATE = 0.05
ALWAYS_UNKNOWN = 1.01  # used when no reachable score meets the target


def unknown_person_paths() -> tuple[list[str], list[str]]:
    """One image per person outside the 62 classes, split into two disjoint halves."""
    paths = []
    for person in lfw_person_dirs():
        images = sorted(str(f) for f in person.iterdir())
        if len(images) < MIN_FACES_PER_PERSON:
            paths.append(images[0])
    order = np.random.RandomState(RANDOM_SEED).permutation(len(paths))
    middle = len(paths) // 2
    return [paths[i] for i in order[:middle]], [paths[i] for i in order[middle:]]


def choose_threshold(unknown_scores: np.ndarray) -> float:
    """The lowest score that lets at most the target share of unknown people through."""
    for score in np.unique(unknown_scores):
        if np.mean(unknown_scores >= score) <= TARGET_UNKNOWN_ACCEPT_RATE:
            return float(score)
    return ALWAYS_UNKNOWN


def known_report(probabilities: np.ndarray, labels: np.ndarray, threshold: float, total: int) -> dict:
    """`total` includes images with no detected face, which count as neither accepted nor right."""
    top = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    accepted = top >= threshold
    return {
        "accuracy": float(correct.sum() / total),
        "accepted_rate": float(accepted.sum() / total),
        "accepted_precision": float(correct[accepted].mean()) if accepted.any() else None,
    }


def cnn_probabilities(model, faces: np.ndarray) -> np.ndarray:
    return model.predict(faces[..., np.newaxis], verbose=0)


def pca_knn_validation(pipeline, data, cropper) -> tuple[np.ndarray, np.ndarray, int]:
    """Out-of-fold scores: each fold is scored by a model that never saw it."""
    faces = crop_faces(data.train_paths, cropper)
    full = np.full((len(data.y_train), len(data.labels)), np.nan)
    positions = np.flatnonzero(faces.detected)
    flat = np.full((len(data.y_train), data.x_train.shape[1]), np.nan)
    flat[positions] = faces.images.reshape(len(positions), -1)

    folds = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    for fit_idx, val_idx in folds.split(data.x_train, data.y_train):
        val_idx = val_idx[faces.detected[val_idx]]
        model = clone(pipeline).fit(data.x_train[fit_idx], data.y_train[fit_idx])
        full[val_idx] = model.predict_proba(flat[val_idx])
    return full[positions], data.y_train[positions], len(data.y_train)


def cnn_validation(model, data, cropper) -> tuple[np.ndarray, np.ndarray, int]:
    # Same arguments as ml.train.train_cnn, so this is the validation split it held out.
    _, val_paths, _, val_y = train_test_split(
        data.train_paths,
        data.y_train,
        test_size=CNN_VAL_SIZE,
        stratify=data.y_train,
        random_state=RANDOM_SEED,
    )
    faces = crop_faces(val_paths, cropper)
    return cnn_probabilities(model, faces.images), val_y[faces.detected], len(val_y)


def cnn_v2_validation(model, data, cropper) -> tuple[np.ndarray, np.ndarray, int]:
    pool = load_pool(data, cropper)
    val_idx = pool.val_idx[pool.detected[pool.val_idx]]
    return cnn_probabilities(model, pool.images[val_idx]), pool.y[val_idx], len(pool.val_idx)


def main() -> None:
    data = load_dataset()
    cropper = FaceCropper()
    calibration_paths, evaluation_paths = unknown_person_paths()
    calibration = crop_faces(calibration_paths, cropper)
    evaluation = crop_faces(evaluation_paths, cropper)
    test_faces = crop_faces(data.test_paths, cropper)
    test_y = data.y_test[test_faces.detected]
    print(
        f"detected: unknown-calibration {calibration.detected.mean():.4f} "
        f"unknown-evaluation {evaluation.detected.mean():.4f} "
        f"test {test_faces.detected.mean():.4f} "
        f"(people: {len(calibration_paths)} + {len(evaluation_paths)})"
    )

    cnn = tf.keras.models.load_model(CNN_DIR / CNN_FILE)
    cnn_v2 = tf.keras.models.load_model(CNN_V2_DIR / CNN_FILE)
    pca_knn = joblib.load(PCA_KNN_DIR / PCA_KNN_FILE)
    models = {
        "cnn": (CNN_DIR, lambda faces: cnn_probabilities(cnn, faces),
                lambda: cnn_validation(cnn, data, cropper)),
        "cnn-v2": (CNN_V2_DIR, lambda faces: cnn_probabilities(cnn_v2, faces),
                   lambda: cnn_v2_validation(cnn_v2, data, cropper)),
        "pca-knn": (PCA_KNN_DIR, lambda faces: pca_knn.predict_proba(faces.reshape(len(faces), -1)),
                    lambda: pca_knn_validation(pca_knn, data, cropper)),
    }

    for name, (directory, predict, validation) in models.items():
        threshold = choose_threshold(predict(calibration.images).max(axis=1))
        val_probabilities, val_y, val_total = validation()
        report = {
            "target_unknown_accept_rate": TARGET_UNKNOWN_ACCEPT_RATE,
            "unknown_accept_rate_evaluation": float(
                np.mean(predict(evaluation.images).max(axis=1) >= threshold)
            ),
            "known_validation": known_report(val_probabilities, val_y, threshold, val_total),
            "known_test": known_report(
                predict(test_faces.images), test_y, threshold, len(data.y_test)
            ),
        }
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            face_crop=DEFAULT_FACE_CROP.to_manifest(),
            unknown_threshold=threshold,
            unknown_calibration=report,
        )
        write_manifest(directory, manifest)
        print(f"{name}: threshold={threshold:.4f} {json.dumps(report)}")


if __name__ == "__main__":
    main()
