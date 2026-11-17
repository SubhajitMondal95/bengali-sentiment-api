
from contextlib import asynccontextmanager

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import model as model_loader
from app.config import MODEL_NAME
from app.predictor import predict_text


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load model when FastAPI starts
@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting Bengali Sentiment API...")

    try:
        model_loader.load_model()
        logger.info("Model loaded successfully.")

    except Exception:
        logger.exception("Failed to load model.")
        raise

    yield

    logger.info("Shutting down Bengali Sentiment API...")


# Initialize FastAPI
app = FastAPI(
    title="Bengali Sentiment Analysis API",
    description="Multi-label Bengali text classification using fine-tuned mBERT.",
    version="1.0.0",
    lifespan=lifespan
)


# Request schema
class PredictionRequest(BaseModel):

    input_text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Bengali sentence for sentiment classification"
    )


# Root endpoint
@app.get("/")
def root():

    return {
        "message": "Bengali Sentiment Analysis API is running.",
        "model": MODEL_NAME,
        "docs": "/docs"
    }


# Health endpoint
@app.get("/health")
def health_check():

    if not model_loader.is_model_loaded():

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded."
        )

    return {
        "status": "healthy",
        "model_loaded": True
    }


# Prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):

    if not model_loader.is_model_loaded():

        raise HTTPException(
            status_code=503,
            detail="Model is not ready."
        )

    try:

        result = predict_text(request.input_text)

        return result

    except Exception:

        logger.exception("Prediction failed.")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please try again."
        )
