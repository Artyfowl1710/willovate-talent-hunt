from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from loguru import logger
import time
from app.schemas import PredictRequest, PredictResponse, HealthResponse, MetricsResponse
from app.model import load_model, run_inference

ml_model = {}
request_stats = {
    "total": 0,
    "success": 0,
    "failed": 0,
    "start_time": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading model...")
    ml_model["instance"] = load_model()
    request_stats["start_time"] = time.time()
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down...")
    ml_model.clear()

app = FastAPI(
    title="Willovate AI/ML API",
    description="Production-grade ML inference API with monitoring",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def root():
    return {"status": "running", "message": "Willovate ML API is live"}

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy",
        model_loaded="instance" in ml_model
    )

@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    return MetricsResponse(
        total_requests=request_stats["total"],
        successful_predictions=request_stats["success"],
        failed_predictions=request_stats["failed"],
        model_loaded="instance" in ml_model
    )

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    request_stats["total"] += 1

    if "instance" not in ml_model:
        request_stats["failed"] += 1
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        prediction, label, confidence = run_inference(
            ml_model["instance"], request.features
        )
        request_stats["success"] += 1
        logger.info("Request #{} — success", request_stats["total"])
        return PredictResponse(
            prediction=prediction,
            label=label,
            confidence=confidence
        )
    except ValueError as e:
        request_stats["failed"] += 1
        logger.warning("Validation error: {}", str(e))
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        request_stats["failed"] += 1
        logger.error("Inference error: {}", str(e))
        raise HTTPException(status_code=500, detail=str(e))