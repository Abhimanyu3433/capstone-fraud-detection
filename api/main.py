"""FastAPI service for real-time online fraud detection.

Run locally:
    uvicorn api.main:app --reload
Swagger UI: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request

from api import azure_client
from api.model_loader import ModelService
from api.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    TransactionRequest,
)
from src.features import INPUT_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("fraud_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the .pkl once at startup, not per request.
    app.state.model = ModelService().load()
    yield


app = FastAPI(
    title="Online Fraud Detection API",
    description="Scores online payment transactions with the tuned Scikit-learn pipeline "
                "(Capstone Project 1, Option A).",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def timing_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-ms"] = f"{ms:.2f}"
    return response


def _model(request: Request) -> ModelService:
    model = getattr(request.app.state, "model", None)
    if model is None or not model.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health(request: Request):
    model = getattr(request.app.state, "model", None)
    loaded = bool(model and model.loaded)
    return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded,
                          model_name=model.model_name if loaded else None,
                          model_version=model.model_version if loaded else None)


@app.get("/model/info", response_model=ModelInfoResponse, tags=["service"])
def model_info(request: Request):
    m = _model(request)
    return ModelInfoResponse(
        model_name=m.model_name, model_version=m.model_version, model_path=str(m.model_path),
        threshold=m.threshold, objective=m.config.get("objective"),
        hyperparameters=m.config.get("hyperparameters", {}),
        test_metrics=m.config.get("test_metrics", {}), input_features=INPUT_FEATURES)


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(txn: TransactionRequest, request: Request):
    """Predict whether one transaction is fraudulent."""
    result = _model(request).predict_records([txn.to_record()])[0]
    log.info("predict type=%s amount=%.2f prob=%.4f fraud=%s",
             txn.type.value, txn.amount, result["fraud_probability"], result["is_fraud"])
    return result


@app.post("/predict/batch", response_model=BatchResponse, tags=["prediction"])
def predict_batch(batch: BatchRequest, request: Request):
    """Score up to 1000 transactions in one call."""
    preds = _model(request).predict_records([t.to_record() for t in batch.transactions])
    return BatchResponse(predictions=preds, count=len(preds),
                         fraud_count=sum(p["prediction"] for p in preds))


@app.post("/predict/azure", response_model=PredictionResponse, tags=["prediction"])
def predict_azure(txn: TransactionRequest, request: Request):
    """Score via the Azure ML managed online endpoint (FastAPI as a client).

    The registered MLflow model applies the decision threshold itself; the threshold
    reported here is the one from best_model.json that the endpoint was built with.
    """
    try:
        r = azure_client.score([txn.to_record()])[0]
    except azure_client.AzureEndpointNotConfigured as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Azure endpoint error: {e}") from e
    except (KeyError, TypeError, ValueError, IndexError) as e:
        raise HTTPException(status_code=502, detail=f"Unexpected Azure response: {e}") from e
    model = getattr(request.app.state, "model", None)
    return PredictionResponse(
        is_fraud=bool(r["is_fraud"]), prediction=int(r["is_fraud"]),
        fraud_probability=r["fraud_probability"],
        threshold=model.threshold if model is not None else 0.5,
        model_name="azureml-endpoint",
        model_version=os.getenv("AZURE_DEPLOYMENT", "default"))
