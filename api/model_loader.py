"""Loads the pickled pipeline once and scores transactions.

Resolution order for the model file:
  1. env MODEL_PATH
  2. artifacts/models/fraud_pipeline.pkl      (full model from Colab)
  3. artifacts/models/fraud_pipeline_ci.pkl   (small CI model, committed to git)
The decision threshold comes from env FRAUD_THRESHOLD or artifacts/best_model.json.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd

from src.features import INPUT_FEATURES

log = logging.getLogger("fraud_api")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))


def _default_model_path() -> Path:
    if os.getenv("MODEL_PATH"):
        return Path(os.environ["MODEL_PATH"])
    full = ARTIFACTS_DIR / "models" / "fraud_pipeline.pkl"
    return full if full.exists() else ARTIFACTS_DIR / "models" / "fraud_pipeline_ci.pkl"


@dataclass
class ModelService:
    model_path: Path = field(default_factory=_default_model_path)
    pipeline: object = None
    threshold: float = 0.5
    model_name: str = "unknown"
    model_version: str = "unknown"
    config: dict = field(default_factory=dict)

    @property
    def loaded(self) -> bool:
        return self.pipeline is not None

    def load(self) -> ModelService:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.pipeline = joblib.load(self.model_path)
        cfg_path = ARTIFACTS_DIR / "best_model.json"
        if cfg_path.exists():
            self.config = json.loads(cfg_path.read_text())
            self.threshold = float(self.config.get("decision_threshold", 0.5))
            self.model_name = self.config.get("model_name", self.model_name)
        if os.getenv("FRAUD_THRESHOLD"):
            self.threshold = float(os.environ["FRAUD_THRESHOLD"])
        self.model_version = os.getenv("MODEL_VERSION", self.model_path.stem)
        log.info("Loaded %s from %s (threshold=%.4f)", self.model_name, self.model_path, self.threshold)
        return self

    def predict_records(self, records: list[dict]) -> list[dict]:
        df = pd.DataFrame.from_records(records)[INPUT_FEATURES]
        proba = self.pipeline.predict_proba(df)[:, 1]
        out = []
        for p in proba:
            label = int(p >= self.threshold)
            out.append({
                "is_fraud": bool(label), "prediction": label, "fraud_probability": float(p),
                "threshold": self.threshold, "model_name": self.model_name,
                "model_version": self.model_version,
            })
        return out
