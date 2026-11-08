from enum import StrEnum

from pydantic import BaseModel


class ModelName(StrEnum):
    cnn = "cnn"
    cnn_v2 = "cnn-v2"
    pca_knn = "pca-knn"


class Prediction(BaseModel):
    name: str
    probability: float


class PredictResponse(BaseModel):
    model: ModelName
    model_version: str
    identity: str  # the top name, or "unknown" when its probability is below the threshold
    predictions: list[Prediction]
    inference_ms: float
    request_id: str


class HealthResponse(BaseModel):
    status: str
    loaded_models: list[ModelName]


class InputSize(BaseModel):
    height: int
    width: int


class ModelSummary(BaseModel):
    model: ModelName
    model_version: str
    num_labels: int
    input_size: InputSize
    test_accuracy: float
    unknown_threshold: float
