import pickle
import numpy as np
import os
from loguru import logger

MODEL_PATH = "model.pkl"
TRAIN_STATS_PATH = "train_stats.csv"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully from {}", MODEL_PATH)
    return model

def validate_input(features: list, expected_length: int = None):
    """Validate input before inference."""
    if not features:
        raise ValueError("Features list is empty")
    if any(np.isnan(v) for v in features):
        raise ValueError("Input contains NaN values")
    if any(np.isinf(v) for v in features):
        raise ValueError("Input contains infinite values")
    if expected_length and len(features) != expected_length:
        raise ValueError(f"Expected {expected_length} features, got {len(features)}")
    logger.info("Input validation passed for {} features", len(features))

def run_inference(model, features: list):
    validate_input(features)

    x = np.array(features).reshape(1, -1)
    prediction = model.predict(x)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        confidence = float(np.max(proba))
        label = str(prediction)
    else:
        confidence = 1.0
        label = str(prediction)

    logger.info("Inference complete — prediction: {}, confidence: {:.4f}", label, confidence)
    return float(prediction), label, confidence