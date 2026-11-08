"""Train the CNN and PCA+KNN models and save them under artifacts/.

Data handling mirrors reference/original_train.py so results stay comparable.
Run with: python -m ml.train
"""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import keras
import numpy as np
import sklearn
import tensorflow as tf
from sklearn.datasets import fetch_lfw_people, get_data_home
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential

from ml.preprocess import DEFAULT_CONFIG

RANDOM_SEED = 0
MODEL_VERSION = "v1"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
CNN_DIR = ARTIFACTS_DIR / f"cnn-{MODEL_VERSION}"
PCA_KNN_DIR = ARTIFACTS_DIR / f"pca-knn-{MODEL_VERSION}"
CNN_FILE = "model.keras"
PCA_KNN_FILE = "model.joblib"

REQUIRED_CLASSES = 62
IMAGES_PER_CLASS = 20
MIN_FACES_PER_PERSON = 20
FUNNELED = True

TEST_SIZE = 0.25
CNN_VAL_SIZE = 0.15
CNN_EPOCHS = 40
CNN_BATCH_SIZE = 32

# k=1 is excluded: its probabilities are only 0 or 1, so a top-3 is meaningless.
PCA_COMPONENTS = [10, 30, 50, 75, 100, 150, 200, 250, 300]
PCA_WHITEN = [False, True]
KNN_NEIGHBORS = [3, 5]
CV_FOLDS = 5


@dataclass(frozen=True)
class Dataset:
    x_train: np.ndarray  # flattened, for PCA+KNN
    x_test: np.ndarray
    img_train: np.ndarray  # (N, h, w), for the CNN
    img_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    train_paths: np.ndarray  # source JPGs
    test_paths: np.ndarray
    labels: list[str]
    height: int
    width: int
    pixel_scale: float


def fetch_lfw():
    return fetch_lfw_people(
        min_faces_per_person=MIN_FACES_PER_PERSON,
        resize=DEFAULT_CONFIG.resize,
        funneled=FUNNELED,
        color=False,
        slice_=DEFAULT_CONFIG.slices,
    )


def lfw_person_dirs() -> list[Path]:
    root = Path(get_data_home()) / "lfw_home" / "lfw_funneled"
    return sorted(p for p in root.iterdir() if p.is_dir())


def lfw_image_paths() -> np.ndarray:
    """JPG paths in the same order as the arrays from fetch_lfw().

    Follows sklearn's _fetch_lfw_people: sorted scan, then a fixed shuffle.
    """
    paths = []
    for person in lfw_person_dirs():
        images = sorted(str(f) for f in person.iterdir())
        if len(images) >= MIN_FACES_PER_PERSON:
            paths.extend(images)
    order = np.arange(len(paths))
    np.random.RandomState(42).shuffle(order)
    return np.array(paths)[order]


def load_dataset() -> Dataset:
    lfw = fetch_lfw()
    n_classes = len(lfw.target_names)
    if n_classes != REQUIRED_CLASSES:
        raise ValueError(f"expected {REQUIRED_CLASSES} classes, found {n_classes}")

    rng = np.random.RandomState(RANDOM_SEED)
    indices = []
    for label in range(n_classes):
        candidates = np.where(lfw.target == label)[0]
        indices.extend(rng.choice(candidates, size=IMAGES_PER_CLASS, replace=False))
    indices = np.array(indices)

    y = np.repeat(np.arange(n_classes), IMAGES_PER_CLASS)
    images = lfw.images[indices]
    pixel_scale = 255.0 if np.max(images) > 1.0 else 1.0

    x_train, x_test, y_train, y_test, img_train, img_test, train_paths, test_paths = train_test_split(
        lfw.data[indices] / pixel_scale,
        y,
        images / pixel_scale,
        lfw_image_paths()[indices],
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )
    return Dataset(
        x_train=x_train,
        x_test=x_test,
        img_train=img_train,
        img_test=img_test,
        y_train=y_train,
        y_test=y_test,
        train_paths=train_paths,
        test_paths=test_paths,
        labels=[str(name) for name in lfw.target_names],
        height=images.shape[1],
        width=images.shape[2],
        pixel_scale=pixel_scale,
    )


def create_cnn(input_shape: tuple[int, int, int], num_classes: int) -> Sequential:
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def cnn_accuracy(model, data: Dataset) -> float:
    images = data.img_test.reshape(-1, data.height, data.width, 1)
    predicted = np.argmax(model.predict(images, verbose=0), axis=1)
    return float(np.mean(predicted == data.y_test))


def pca_knn_accuracy(model, data: Dataset) -> float:
    return float(np.mean(model.predict(data.x_test) == data.y_test))


def train_cnn(data: Dataset) -> Sequential:
    img_train, img_val, y_train, y_val = train_test_split(
        data.img_train,
        data.y_train,
        test_size=CNN_VAL_SIZE,
        stratify=data.y_train,
        random_state=RANDOM_SEED,
    )
    h, w = data.height, data.width
    model = create_cnn((h, w, 1), len(data.labels))
    model.fit(
        img_train.reshape(-1, h, w, 1),
        y_train,
        epochs=CNN_EPOCHS,
        batch_size=CNN_BATCH_SIZE,
        validation_data=(img_val.reshape(-1, h, w, 1), y_val),
        verbose=2,
    )
    return model


def train_pca_knn(data: Dataset) -> GridSearchCV:
    """Pick the configuration by CV inside train; test is never consulted."""
    pipeline = Pipeline(
        [
            ("pca", PCA(random_state=RANDOM_SEED)),
            ("knn", KNeighborsClassifier()),
        ]
    )
    search = GridSearchCV(
        pipeline,
        param_grid={
            "pca__n_components": PCA_COMPONENTS,
            "pca__whiten": PCA_WHITEN,
            "knn__n_neighbors": KNN_NEIGHBORS,
        },
        cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
        scoring="accuracy",
    )
    return search.fit(data.x_train, data.y_train)


def build_manifest(model_name: str, test_accuracy: float, data: Dataset, **extra) -> dict:
    return {
        "model": model_name,
        "model_version": MODEL_VERSION,
        "labels": data.labels,
        "input_size": {"height": data.height, "width": data.width},
        "preprocessing": {
            "funneled": FUNNELED,
            **DEFAULT_CONFIG.to_manifest(),
            "pixel_scale": data.pixel_scale,
        },
        "test_accuracy": test_accuracy,
        "test_samples": len(data.y_test),
        "versions": {
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "sklearn": sklearn.__version__,
        },
        "seed": RANDOM_SEED,
        **extra,
    }


def write_manifest(directory: Path, manifest: dict) -> None:
    text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    (directory / "manifest.json").write_text(text, encoding="utf-8")


def main() -> None:
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    tf.config.experimental.enable_op_determinism()

    data = load_dataset()
    print(f"train={len(data.y_train)} test={len(data.y_test)} classes={len(data.labels)}")

    search = train_pca_knn(data)
    pca_knn = search.best_estimator_
    pca_knn_acc = pca_knn_accuracy(pca_knn, data)
    PCA_KNN_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca_knn, PCA_KNN_DIR / PCA_KNN_FILE)
    write_manifest(
        PCA_KNN_DIR,
        build_manifest(
            "pca-knn",
            pca_knn_acc,
            data,
            best_params=search.best_params_,
            cv_accuracy=float(search.best_score_),
        ),
    )
    print(f"pca-knn test={pca_knn_acc:.4f} cv={search.best_score_:.4f} {search.best_params_}")

    cnn = train_cnn(data)
    cnn_acc = cnn_accuracy(cnn, data)
    CNN_DIR.mkdir(parents=True, exist_ok=True)
    cnn.save(CNN_DIR / CNN_FILE)
    write_manifest(CNN_DIR, build_manifest("cnn", cnn_acc, data))
    print(f"cnn test={cnn_acc:.4f}")


if __name__ == "__main__":
    main()
