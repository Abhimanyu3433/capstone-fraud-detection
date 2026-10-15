"""Test fixtures.

Tests use the small CI model (artifacts/models/fraud_pipeline_ci.pkl) produced by
the Colab notebook. If the Colab hand-off has not been copied in yet, a tiny
stand-in model is trained on synthetic data so the suite (and CI) still runs.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_ARTIFACTS = PROJECT_ROOT / "artifacts"
CI_MODEL = REAL_ARTIFACTS / "models" / "fraud_pipeline_ci.pkl"


def synthetic_transactions(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """PaySim-like rows: frauds are TRANSFER/CASH_OUT that drain the sender balance."""
    rng = np.random.default_rng(seed)
    types = rng.choice(["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"], n,
                       p=[0.22, 0.35, 0.01, 0.34, 0.08])
    old = np.round(rng.lognormal(10, 2, n), 2) * (rng.random(n) > 0.3)
    amount = np.round(rng.lognormal(9, 1.2, n), 2) + 1
    fraud = (np.isin(types, ["TRANSFER", "CASH_OUT"]) & (rng.random(n) < 0.08) & (old > 0)).astype(int)
    amount = np.where(fraud == 1, old, amount)
    new = np.where(fraud == 1, 0.0, np.maximum(old - amount, 0))
    od = np.round(rng.lognormal(11, 2, n), 2) * (rng.random(n) > 0.4)
    nd = od + np.where(fraud == 1, 0.0, amount)
    return pd.DataFrame({"step": rng.integers(1, 744, n), "type": types, "amount": amount,
                         "oldbalanceOrg": old, "newbalanceOrig": new, "oldbalanceDest": od,
                         "newbalanceDest": nd, "Is_Fraud": fraud})


def _build_standin(art_dir: Path) -> None:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from src.features import INPUT_FEATURES, build_preprocessor

    df = synthetic_transactions()
    pipe = Pipeline([("prep", build_preprocessor()),
                     ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    pipe.fit(df[INPUT_FEATURES], df["Is_Fraud"])
    (art_dir / "models").mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, art_dir / "models" / "fraud_pipeline_ci.pkl")
    (art_dir / "best_model.json").write_text(json.dumps({
        "model_key": "logreg", "model_name": "Logistic Regression (test stand-in)",
        "estimator_class": "sklearn.linear_model._logistic.LogisticRegression",
        "hyperparameters": {"max_iter": 1000, "class_weight": "balanced"},
        "imbalance": {"strategy": "class_weight", "sampler": None, "sampler_ratio": None},
        "objective": "f1", "decision_threshold": 0.5, "test_metrics": {}, "validation_metrics": {},
        "pipeline_file": "models/fraud_pipeline_ci.pkl", "ci_pipeline_file": "models/fraud_pipeline_ci.pkl",
    }))


def pytest_configure(config):
    # Must run before `api` is imported: the loader reads these at import time.
    if CI_MODEL.exists() and (REAL_ARTIFACTS / "best_model.json").exists():
        os.environ.setdefault("ARTIFACTS_DIR", str(REAL_ARTIFACTS))
        os.environ.setdefault("MODEL_PATH", str(CI_MODEL))
    else:
        tmp = Path(tempfile.mkdtemp(prefix="fraud_artifacts_"))
        _build_standin(tmp)
        os.environ["ARTIFACTS_DIR"] = str(tmp)
        os.environ["MODEL_PATH"] = str(tmp / "models" / "fraud_pipeline_ci.pkl")
    for var in ("AZURE_ENDPOINT_URL", "AZURE_ENDPOINT_KEY", "AZURE_DEPLOYMENT", "FRAUD_THRESHOLD"):
        os.environ.pop(var, None)


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as c:  # runs the lifespan -> model loaded once
        yield c


@pytest.fixture
def valid_txn() -> dict:
    return {"step": 410, "type": "TRANSFER", "amount": 181000.0, "oldbalanceOrg": 181000.0,
            "newbalanceOrig": 0.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0}


@pytest.fixture
def legit_txn() -> dict:
    return {"step": 12, "type": "PAYMENT", "amount": 9839.64, "oldbalanceOrg": 170136.0,
            "newbalanceOrig": 160296.36, "oldbalanceDest": 0.0, "newbalanceDest": 0.0}


@pytest.fixture(scope="session")
def artifacts_dir() -> Path:
    return Path(os.environ["ARTIFACTS_DIR"])
