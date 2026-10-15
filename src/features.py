"""Shared feature engineering for the online fraud detection project.

This module is imported by BOTH the Colab model-selection notebook and the
VS Code application (training script, FastAPI, tests). Every object that ends
up inside the pickled pipeline is defined here, so the .pkl file can be loaded
anywhere `src.features` is importable.

Dataset: onlinefraud.csv (PaySim-style simulation, ~6.36M rows).
Raw columns: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
             nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------
# Column definitions
# --------------------------------------------------------------------------
TARGET = "Is_Fraud"

# Renames applied right after loading (brief names the target `Is_Fraud`).
RENAME_MAP = {"isFraud": TARGET, "isFlaggedFraud": "Is_Flagged_Fraud"}

# Identifier columns: near-unique per row, not available as stable signal.
ID_COLUMNS = ["nameOrig", "nameDest"]

# Rule-based flag produced by the source system; leaks the label -> dropped.
LEAKY_COLUMNS = ["Is_Flagged_Fraud"]

# Fields a client sends to the API (one transaction, no history needed).
RAW_NUMERIC = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
RAW_CATEGORICAL = ["type"]
INPUT_FEATURES = RAW_NUMERIC + RAW_CATEGORICAL

TYPE_CATEGORIES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# Money-valued columns (heavy-tailed) -> signed log transform before scaling.
MONEY_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "error_balance_orig",
    "error_balance_dest",
]
# Bounded / count-like engineered features -> scaled only.
OTHER_NUMERIC_FEATURES = [
    "hour",
    "is_night",
    "orig_zero_after",
    "orig_zero_before",
    "dest_zero_before",
    "amount_to_orig_balance",
    "is_full_drain",
]
ENGINEERED_FEATURES = [
    "hour",
    "is_night",
    "error_balance_orig",
    "error_balance_dest",
    "orig_zero_after",
    "orig_zero_before",
    "dest_zero_before",
    "amount_to_orig_balance",
    "is_full_drain",
]


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename source columns to project names (idempotent)."""
    return df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})


def signed_log1p(x):
    """sign(x) * log(1 + |x|): compresses heavy tails, keeps negative values."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


# --------------------------------------------------------------------------
# Domain feature engineering
# --------------------------------------------------------------------------
class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds per-transaction domain features computable from a single request.

    - hour / is_night: `step` is one hour of simulated time. The absolute day
      index is NOT used as a feature: with a time-based split it would only
      encode position in time and cannot generalise to future data.
    - error_balance_orig/dest: balance discrepancy (brief's suggested feature):
        orig: newbalanceOrig + amount - oldbalanceOrg   (0 when books balance)
        dest: oldbalanceDest + amount - newbalanceDest
    - zero-balance flags, amount relative to the sender's balance, full-drain flag.

    Velocity features (transactions per account per window) are deliberately
    not used: nameOrig is almost unique per row and per-account history is not
    available at real-time scoring time. This is documented in the report.
    """

    night_hours = (0, 6)  # [start, end) in hours

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        hour = df["step"].astype(int) % 24
        df["hour"] = hour
        df["is_night"] = ((hour >= self.night_hours[0]) & (hour < self.night_hours[1])).astype(int)
        df["error_balance_orig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
        df["error_balance_dest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
        df["orig_zero_after"] = (df["newbalanceOrig"] == 0).astype(int)
        df["orig_zero_before"] = (df["oldbalanceOrg"] == 0).astype(int)
        df["dest_zero_before"] = (df["oldbalanceDest"] == 0).astype(int)
        df["amount_to_orig_balance"] = np.clip(
            df["amount"] / (df["oldbalanceOrg"] + 1.0), 0, 1e3
        )
        df["is_full_drain"] = (
            (df["oldbalanceOrg"] > 0) & (np.isclose(df["amount"], df["oldbalanceOrg"]))
        ).astype(int)
        keep = RAW_CATEGORICAL + list(dict.fromkeys(MONEY_FEATURES + OTHER_NUMERIC_FEATURES))
        return df[keep]

    def get_feature_names_out(self, input_features=None):
        return np.array(
            RAW_CATEGORICAL + list(dict.fromkeys(MONEY_FEATURES + OTHER_NUMERIC_FEATURES)),
            dtype=object,
        )


def _column_transformer() -> ColumnTransformer:
    money = Pipeline(
        [
            ("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("money", money, MONEY_FEATURES),
            ("num", StandardScaler(), OTHER_NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(categories=[TYPE_CATEGORIES], handle_unknown="ignore"),
                RAW_CATEGORICAL,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


class FraudPreprocessor(BaseEstimator, TransformerMixin):
    """Feature engineering + encoding + scaling as ONE estimator.

    imbalanced-learn pipelines reject nested sklearn Pipelines as intermediate
    steps, so the engineer and the ColumnTransformer are wrapped here instead.
    """

    def fit(self, X, y=None):
        self.engineer_ = FraudFeatureEngineer()
        self.columns_ = _column_transformer().fit(self.engineer_.transform(X))
        return self

    def transform(self, X):
        return self.columns_.transform(self.engineer_.transform(X))

    def get_feature_names_out(self, input_features=None):
        return self.columns_.get_feature_names_out()


def build_preprocessor() -> FraudPreprocessor:
    """Feature engineering + encoding + scaling (unfitted)."""
    return FraudPreprocessor()
