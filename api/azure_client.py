"""Client for the Azure ML managed online endpoint (FastAPI acts as a client).

Configure with environment variables (never commit keys):
    AZURE_ENDPOINT_URL   scoring URI, e.g. https://fraud-endpoint.<region>.inference.ml.azure.com/score
    AZURE_ENDPOINT_KEY   primary key of the endpoint
    AZURE_DEPLOYMENT     optional: force a deployment (e.g. "green") via azureml-model-deployment header
"""
from __future__ import annotations

import os

import httpx

from src.features import INPUT_FEATURES


class AzureEndpointNotConfigured(RuntimeError):
    pass


def is_configured() -> bool:
    return bool(os.getenv("AZURE_ENDPOINT_URL") and os.getenv("AZURE_ENDPOINT_KEY"))


def build_payload(records: list[dict]) -> dict:
    """MLflow 'input_data' split-orientation payload expected by Azure ML for MLflow models."""
    return {"input_data": {"columns": INPUT_FEATURES, "index": list(range(len(records))),
                           "data": [[r[c] for c in INPUT_FEATURES] for r in records]}}


def parse_response(body) -> list[dict]:
    """Normalise the endpoint output to [{'fraud_probability': float, 'is_fraud': int}, ...]."""
    if isinstance(body, dict):
        body = body.get("predictions", body.get("output", body))
    if isinstance(body, dict) and "fraud_probability" in body:  # column-oriented
        keys = list(body)
        body = [dict(zip(keys, vals, strict=True)) for vals in zip(*body.values(), strict=True)]
    return [{"fraud_probability": float(r["fraud_probability"]), "is_fraud": int(r["is_fraud"])} for r in body]


def score(records: list[dict], timeout: float = 30.0) -> list[dict]:
    if not is_configured():
        raise AzureEndpointNotConfigured("Set AZURE_ENDPOINT_URL and AZURE_ENDPOINT_KEY")
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {os.environ['AZURE_ENDPOINT_KEY']}"}
    if os.getenv("AZURE_DEPLOYMENT"):
        headers["azureml-model-deployment"] = os.environ["AZURE_DEPLOYMENT"]
    resp = httpx.post(os.environ["AZURE_ENDPOINT_URL"], json=build_payload(records),
                      headers=headers, timeout=timeout)
    resp.raise_for_status()
    return parse_response(resp.json())
