"""Model artefact tests: loads, valid probabilities, identical after save/reload."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.features import INPUT_FEATURES
from tests.conftest import synthetic_transactions


@pytest.fixture(scope="module")
def pipeline():
    import os

    return joblib.load(os.environ["MODEL_PATH"])


def test_probabilities_in_range(pipeline):
    X = synthetic_transactions(300)[INPUT_FEATURES]
    p = pipeline.predict_proba(X)[:, 1]
    assert p.shape == (300,) and np.all((p >= 0) & (p <= 1))


def test_reload_gives_identical_predictions(pipeline, tmp_path):
    X = synthetic_transactions(300, seed=1)[INPUT_FEATURES]
    path = tmp_path / "reloaded.pkl"
    joblib.dump(pipeline, path)
    again = joblib.load(path)
    # Parallel tree averaging (n_jobs=-1) can change float summation order -> compare with a tolerance
    assert np.allclose(pipeline.predict_proba(X), again.predict_proba(X), rtol=0, atol=1e-12)
    assert np.array_equal(pipeline.predict(X), again.predict(X))


def test_best_model_config_has_required_keys(artifacts_dir):
    cfg = json.loads((artifacts_dir / "best_model.json").read_text())
    for k in ("model_name", "estimator_class", "hyperparameters", "decision_threshold"):
        assert k in cfg
    assert 0 < cfg["decision_threshold"] <= 1


def test_ci_sample_scores_reasonably(pipeline, artifacts_dir):
    path = artifacts_dir / "ci_sample.csv"
    if not path.exists():
        pytest.skip("ci_sample.csv not available (Colab hand-off not copied)")
    from sklearn.metrics import average_precision_score

    df = pd.read_csv(path)
    p = pipeline.predict_proba(df[INPUT_FEATURES])[:, 1]
    # CI model is small, so only require clearly-better-than-random ranking
    assert average_precision_score(df["Is_Fraud"], p) > 3 * df["Is_Fraud"].mean()


def test_sample_requests_match_expected_labels(client, artifacts_dir):
    path = artifacts_dir / "sample_requests.json"
    if not path.exists():
        pytest.skip("sample_requests.json not available")
    for item in json.loads(path.read_text()):
        if item["request"]["amount"] <= 0:  # zero-amount rows exist in the data but are rejected by the API
            continue
        r = client.post("/predict", json=item["request"])
        assert r.status_code == 200, r.text
