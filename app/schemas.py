from pydantic import BaseModel, field_validator
from typing import List

class PredictRequest(BaseModel):
    features: List[float]

    @field_validator("features")
    @classmethod
    def features_must_not_be_empty(cls, v):
        if len(v) == 0:
            raise ValueError("Features list cannot be empty")
        return v

class PredictResponse(BaseModel):
    prediction: float
    label: str
    confidence: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

class MetricsResponse(BaseModel):
    total_requests: int
    successful_predictions: int
    failed_predictions: int
    model_loaded: bool