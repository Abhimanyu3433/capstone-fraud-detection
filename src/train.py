"""Rebuild the selected model from the Colab hand-off and (optionally) log it to MLflow.

The model type, hyperparameters and imbalance strategy come from
artifacts/best_model.json, so no tuning happens here: this is a single,
time-bounded fit. Use it for retraining (scheduled / drift-triggered), the
optional CD workflow, or an Azure ML command job.

Examples
    # quick local rebuild on a subset (laptop-friendly)
    python -m src.train --data data/raw/onlinefraud.csv --max-rows 1500000
    # same, logging the run to MLflow (Azure tracking URI or local ./mlruns)
    python -m src.train --data data/raw/onlinefraud.csv --max-rows 1500000 --mlflow
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import joblib

from src.config import ARTIFACTS_DIR, PROJECT_ROOT, load_best_model_config
from src.data import clean, dev_sample, load_raw, time_split, xy
from src.evaluate import f1_optimal_threshold, metrics_at
from src.features import build_preprocessor


def make_sampler(imbalance: dict, random_state: int):
    sampler = imbalance.get("sampler")
    if not sampler:
        return None
    ratio = imbalance.get("sampler_ratio") or 0.5
    if sampler == "SMOTE":
        from imblearn.over_sampling import SMOTE
        return SMOTE(sampling_strategy=ratio, random_state=random_state)
    if sampler == "RandomOverSampler":
        from imblearn.over_sampling import RandomOverSampler
        return RandomOverSampler(sampling_strategy=ratio, random_state=random_state)
    if sampler == "RandomUnderSampler":
        from imblearn.under_sampling import RandomUnderSampler
        return RandomUnderSampler(sampling_strategy=ratio, random_state=random_state)
    raise ValueError(f"Unknown sampler {sampler}")


def build_pipeline_from_config(cfg=None, overrides: dict | None = None):
    """Unfitted pipeline (prep -> [sampler] -> clf) matching the Colab winner."""
    cfg = cfg or load_best_model_config()
    module, cls_name = cfg.estimator_class.rsplit(".", 1)
    cls = getattr(importlib.import_module(module), cls_name)
    params = {**cfg.hyperparameters, **(overrides or {})}
    rs = cfg.raw.get("random_state", 42)
    if "random_state" in cls().get_params():
        params.setdefault("random_state", rs)
    clf = cls(**params)
    sampler = make_sampler(cfg.imbalance, rs)
    if sampler is None:
        from sklearn.pipeline import Pipeline
        return Pipeline([("prep", build_preprocessor()), ("clf", clf)])
    from imblearn.pipeline import Pipeline as ImbPipeline
    return ImbPipeline([("prep", build_preprocessor()), ("sampler", sampler), ("clf", clf)])


def train(data_path: str, max_rows: int | None, out: Path, neg_pos_ratio: int | None = None,
          use_mlflow: bool = False, experiment: str = "fraud-detection") -> dict:
    cfg = load_best_model_config()
    t0 = time.time()
    df = clean(load_raw(data_path, nrows=max_rows))
    train_df, val_df, test_df = time_split(df)
    ratio = neg_pos_ratio or cfg.raw.get("training_data", {}).get("neg_pos_ratio", 20)
    dev = dev_sample(train_df, ratio, cfg.raw.get("random_state", 42))
    X_dev, y_dev = xy(dev)
    X_val, y_val = xy(val_df)
    X_test, y_test = xy(test_df)

    pipe = build_pipeline_from_config(cfg)
    pipe.fit(X_dev, y_dev)
    threshold = f1_optimal_threshold(y_val, pipe.predict_proba(X_val)[:, 1])
    test_proba = pipe.predict_proba(X_test)[:, 1]
    test_m = metrics_at(y_test, test_proba, threshold)
    secs = round(time.time() - t0, 1)

    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out, compress=3)
    summary = {
        "model": cfg.model_name, "hyperparameters": cfg.hyperparameters,
        "imbalance": cfg.imbalance, "neg_pos_ratio": ratio, "rows_used": len(df),
        "dev_rows": len(dev), "decision_threshold": threshold, "test_metrics": test_m,
        "train_seconds": secs, "pipeline_file": str(out),
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "hyperparameters"}, indent=2))

    if use_mlflow:
        from mlops.mlflow_utils import log_retrain_run
        log_retrain_run(pipe, summary, X_test, y_test, test_proba, experiment)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(PROJECT_ROOT / "data" / "raw" / "onlinefraud.csv"))
    ap.add_argument("--max-rows", type=int, default=None,
                    help="read only the first N rows (earliest steps) to keep training short")
    ap.add_argument("--neg-pos-ratio", type=int, default=None)
    ap.add_argument("--out", default=str(ARTIFACTS_DIR / "models" / "fraud_pipeline_retrained.pkl"))
    ap.add_argument("--mlflow", action="store_true", help="log the run to MLflow")
    ap.add_argument("--experiment", default="fraud-detection")
    a = ap.parse_args()
    train(a.data, a.max_rows, Path(a.out), a.neg_pos_ratio, a.mlflow, a.experiment)


if __name__ == "__main__":
    main()
