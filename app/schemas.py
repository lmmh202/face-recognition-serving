from enum import StrEnum

from pydantic import BaseModel


class ModelName(StrEnum):
    cnn = "cnn"
    pca_knn = "pca-knn"


class Prediction(BaseModel):
    name: str
    probability: float


class PredictResponse(BaseModel):
    model: ModelName
    model_version: str
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
