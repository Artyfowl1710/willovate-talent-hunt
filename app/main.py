from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.schemas import PredictRequest, PredictResponse
from app.model import load_model, run_inference

ml_model = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    ml_model["instance"] = load_model()
    print("Model loaded successfully!")
    yield
    ml_model.clear()

app = FastAPI(
    title="Willovate AI/ML API",
    description="Production-grade ML inference API",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def root():
    return {"status": "running", "message": "Willovate ML API is live"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": "instance" in ml_model}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if "instance" not in ml_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        prediction, label, confidence = run_inference(ml_model["instance"], request.features)
        return PredictResponse(
            prediction=prediction,
            label=label,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))