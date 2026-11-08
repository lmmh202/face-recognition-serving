import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import get_data_home

from app.main import app
from app.registry import DEFAULT_ARTIFACTS_DIR


@pytest.fixture(scope="session")
def client():
    root = Path(os.environ.get("ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR))
    required = [root / "cnn-v1" / "model.keras", root / "pca-knn-v1" / "model.joblib"]
    if not all(path.exists() for path in required):
        pytest.skip("model artifacts not found; run python -m ml.train")
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def require_lfw():
    if not (Path(get_data_home()) / "lfw_home" / "lfw_funneled").is_dir():
        pytest.skip("LFW dataset not found")
