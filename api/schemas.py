"""Pydantic request/response models for the fraud detection API.

Field ranges follow the training data (artifacts/feature_spec.json) with
headroom: money values must be non-negative and below 1e10, `step` is an hour
index >= 1, and `type` must be one of the five transaction types seen in
training (any other value is rejected with HTTP 422).
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_MONEY = 1e10
MAX_BATCH = 1000


class TransactionType(str, Enum):
    CASH_IN = "CASH_IN"
    CASH_OUT = "CASH_OUT"
    DEBIT = "DEBIT"
    PAYMENT = "PAYMENT"
    TRANSFER = "TRANSFER"


class TransactionRequest(BaseModel):
    """One online transaction to score."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "step": 410,
                "type": "TRANSFER",
                "amount": 181000.0,
                "oldbalanceOrg": 181000.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
            }
        },
    )

    step: int = Field(..., ge=1, le=100_000, description="Hour index of the transaction (1 = first hour)")
    type: TransactionType = Field(..., description="Transaction type")
    amount: float = Field(..., gt=0, le=MAX_MONEY, description="Transaction amount")
    oldbalanceOrg: float = Field(..., ge=0, le=MAX_MONEY, description="Sender balance before")
    newbalanceOrig: float = Field(..., ge=0, le=MAX_MONEY, description="Sender balance after")
    oldbalanceDest: float = Field(..., ge=0, le=MAX_MONEY, description="Recipient balance before")
    newbalanceDest: float = Field(..., ge=0, le=MAX_MONEY, description="Recipient balance after")

    @model_validator(mode="after")
    def _no_nan(self):
        for name in ("amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"):
            v = getattr(self, name)
            if v != v:  # NaN check
                raise ValueError(f"{name} must be a number")
        return self

    def to_record(self) -> dict:
        d = self.model_dump()
        d["type"] = self.type.value
        return d


class PredictionResponse(BaseModel):
    is_fraud: bool
    prediction: int = Field(..., description="1 = fraud, 0 = legitimate")
    fraud_probability: float = Field(..., ge=0, le=1)
    threshold: float
    model_name: str
    model_version: str


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transactions: list[TransactionRequest] = Field(..., min_length=1, max_length=MAX_BATCH)


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int
    fraud_count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_path: str
    threshold: float
    objective: str | None = None
    hyperparameters: dict = {}
    test_metrics: dict = {}
    input_features: list[str]
