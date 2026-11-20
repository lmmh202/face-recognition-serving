import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import get_data_home

from app.main import app
from app.registry import DEFAULT_ARTIFACTS_DIR

# CI sets this so that missing data fails the run instead of silently skipping
# the golden, accuracy, and API tests.
REQUIRE_TEST_DATA = os.environ.get("REQUIRE_TEST_DATA") == "1"


def skip_or_fail(reason: str) -> None:
    if REQUIRE_TEST_DATA:
        pytest.fail(f"{reason} (REQUIRE_TEST_DATA=1)", pytrace=False)
    pytest.skip(reason)


@pytest.fixture(scope="session")
def client():
    root = Path(os.environ.get("ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR))
    required = [
        root / "cnn-v1" / "model.keras",
        root / "cnn-v2" / "model.keras",
        root / "pca-knn-v1" / "model.joblib",
    ]
    if not all(path.exists() for path in required):
        skip_or_fail("model artifacts not found; run python -m ml.train")
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def require_lfw():
    if not (Path(get_data_home()) / "lfw_home" / "lfw_funneled").is_dir():
        skip_or_fail("LFW dataset not found")
