"""Unit tests for the shared feature engineering module."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import (
    INPUT_FEATURES,
    FraudFeatureEngineer,
    build_preprocessor,
    signed_log1p,
    standardize_columns,
)
from tests.conftest import synthetic_transactions


def _row(**kw):
    base = {"step": 25, "type": "TRANSFER", "amount": 100.0, "oldbalanceOrg": 100.0,
            "newbalanceOrig": 0.0, "oldbalanceDest": 50.0, "newbalanceDest": 150.0}
    base.update(kw)
    return pd.DataFrame([base])


def test_standardize_columns_renames_target():
    df = standardize_columns(pd.DataFrame({"isFraud": [1], "isFlaggedFraud": [0]}))
    assert list(df.columns) == ["Is_Fraud", "Is_Flagged_Fraud"]


def test_signed_log1p_keeps_sign_and_zero():
    out = signed_log1p(np.array([-np.e + 1, 0.0, np.e - 1]))
    assert np.allclose(out, [-1, 0, 1])


def test_engineered_features():
    out = FraudFeatureEngineer().transform(_row())
    r = out.iloc[0]
    assert r["hour"] == 1 and r["is_night"] == 1
    assert r["error_balance_orig"] == 0.0          # 0 + 100 - 100
    assert r["error_balance_dest"] == 0.0          # 50 + 100 - 150
    assert r["orig_zero_after"] == 1 and r["is_full_drain"] == 1
    assert "step" not in out.columns               # raw step replaced by hour


def test_balance_discrepancy_detected():
    r = FraudFeatureEngineer().transform(_row(newbalanceDest=50.0)).iloc[0]
    assert r["error_balance_dest"] == 100.0


def test_day_time_flags():
    r = FraudFeatureEngineer().transform(_row(step=14)).iloc[0]
    assert r["hour"] == 14 and r["is_night"] == 0


def test_preprocessor_output_shape_and_no_nans():
    df = synthetic_transactions(500)
    prep = build_preprocessor().fit(df[INPUT_FEATURES])
    X = prep.transform(df[INPUT_FEATURES])
    names = prep.get_feature_names_out()
    assert X.shape == (500, len(names))
    assert not np.isnan(np.asarray(X, dtype=float)).any()
    assert {"type_TRANSFER", "error_balance_orig", "hour"} <= set(names)


def test_unknown_type_does_not_crash_preprocessor():
    df = synthetic_transactions(200)
    prep = build_preprocessor().fit(df[INPUT_FEATURES])
    X = prep.transform(_row(type="UNKNOWN")[INPUT_FEATURES])
    assert X.shape[0] == 1
