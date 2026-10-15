"""Data loading, cleaning and splitting (same logic as the Colab notebook).

Used by `src.train` for local / CI / Azure retraining. The full notebook remains
the reference implementation for EDA and model selection.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features import ID_COLUMNS, INPUT_FEATURES, LEAKY_COLUMNS, TARGET, standardize_columns

RAW_DTYPES = {
    "step": "int16",
    "type": "category",
    "amount": "float64",
    "nameOrig": "string",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "nameDest": "string",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}


def load_raw(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Read the onlinefraud CSV with memory-friendly dtypes."""
    return pd.read_csv(path, dtype=RAW_DTYPES, nrows=nrows)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop missing/duplicate rows, rename the target, drop identifier and leaky columns."""
    df = df.dropna().drop_duplicates()
    df = standardize_columns(df)
    return df.drop(columns=[c for c in ID_COLUMNS + LEAKY_COLUMNS if c in df.columns])


def time_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15):
    """Split by `step` (time) into train / validation / test without splitting a step."""
    df = df.sort_values("step", kind="stable").reset_index(drop=True)
    n = len(df)
    train_end = int(df["step"].iloc[int(n * train_frac)])
    val_end = int(df["step"].iloc[int(n * (train_frac + val_frac))])
    train = df[df["step"] <= train_end]
    val = df[(df["step"] > train_end) & (df["step"] <= val_end)]
    test = df[df["step"] > val_end]
    return train, val, test


def dev_sample(train: pd.DataFrame, neg_pos_ratio: int, random_state: int = 42) -> pd.DataFrame:
    """All frauds + `neg_pos_ratio` randomly sampled legitimate rows per fraud."""
    pos = train[train[TARGET] == 1]
    n_neg = min(len(pos) * neg_pos_ratio, int((train[TARGET] == 0).sum()))
    neg = train[train[TARGET] == 0].sample(n=n_neg, random_state=random_state)
    return pd.concat([pos, neg]).sample(frac=1, random_state=random_state).reset_index(drop=True)


def xy(df: pd.DataFrame):
    return df[INPUT_FEATURES], df[TARGET].astype(int)
