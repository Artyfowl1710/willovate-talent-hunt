import pickle
import numpy as np
import os

MODEL_PATH = "model.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

def run_inference(model, features: list):
    x = np.array(features).reshape(1, -1)
    prediction = model.predict(x)[0]
    
    # If classifier, get confidence
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        confidence = float(np.max(proba))
        label = str(prediction)
    else:
        confidence = 1.0  # regression models don't have confidence
        label = str(prediction)
    
    return float(prediction), label, confidence