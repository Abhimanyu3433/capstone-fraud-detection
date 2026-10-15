"""Section 3, Tasks 2-3: MLflow experiment tracking on the Azure ML workspace.

Creates one MLflow run per model compared in Colab (params, CV/validation/test
metrics, confusion matrix, ROC/PR curves) and a final run for the selected
model that also logs the feature-importance plot, EDA figures and the model
itself (pyfunc wrapping the .pkl + decision threshold).

    python -m mlops.track_experiments                 # Azure ML tracking URI
    python -m mlops.track_experiments --local         # dry run into ./mlruns (mlflow ui)
    python -m mlops.track_experiments --register      # also register the best model
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import mlflow
import pandas as pd

from mlops.azure_config import EXPERIMENT_NAME, MODEL_NAME, set_mlflow_tracking
from mlops.mlflow_utils import log_pyfunc_model
from src.config import ARTIFACTS_DIR, load_best_model_config
from src.evaluate import plot_confusion_matrix, plot_roc_pr

METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "tn", "fp", "fn", "tp"]


def _flat_params(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in (d or {}).items():
        out[f"{prefix}{k}"] = json.dumps(v) if isinstance(v, dict | list) else v
    return out


def log_model_runs(summary: dict, curves: dict, best_key: str, tmp: Path) -> dict:
    run_ids = {}
    for key, s in summary.items():
        with mlflow.start_run(run_name=f"{key}-{s['model']}") as run:
            mlflow.set_tags({"model_key": key, "model_family": s["model"], "source": "colab-model-selection",
                             "is_best": str(key == best_key), "objective": "f1"})
            params = {k.replace("clf__", ""): v for k, v in s.get("best_params", {}).items()}
            mlflow.log_params(_flat_params({**params, **s.get("fixed_params", {})}))
            mlflow.log_params({"sampler": s.get("sampler") or "none", "n_iter": s.get("n_iter"),
                               "threshold_f1": round(s["threshold_f1"], 6),
                               "threshold_min_cost": round(s["threshold_cost"], 6)})
            mlflow.log_metrics({k: v for k, v in s.items() if k.startswith("cv_") and isinstance(v, float)})
            for split, m in [("val", s["val_metrics"]), ("test", s["test_metrics_at_threshold"]),
                             ("test_at_0_5", s["test_metrics_at_0_5"])]:
                mlflow.log_metrics({f"{split}_{k}": float(m[k]) for k in METRIC_KEYS})
            mlflow.log_metrics({"latency_ms_single_row": s["latency_ms_single_row"],
                                "search_seconds": s["search_seconds"]})
            cm = plot_confusion_matrix(s["test_metrics_at_threshold"], f"{s['model']} (test)",
                                       tmp / f"{key}_confusion_matrix.png")
            mlflow.log_artifact(cm, "plots")
            if key in curves:
                c = curves[key]
                mlflow.log_artifact(plot_roc_pr(c["roc"]["fpr"], c["roc"]["tpr"], c["pr"]["precision"],
                                                c["pr"]["recall"], f"{s['model']} (test)",
                                                tmp / f"{key}_roc_pr.png", c.get("prevalence")), "plots")
            cv_csv = ARTIFACTS_DIR / "cv_results" / f"{key}.csv"
            if cv_csv.exists():
                mlflow.log_artifact(str(cv_csv), "cv_results")
            run_ids[key] = run.info.run_id
            print(f"logged {key}: run {run.info.run_id}  test F1={s['test_metrics_at_threshold']['f1']:.4f}")
    return run_ids


def log_best_run(cfg, use_ci_model: bool, register: bool) -> str:
    pkl = cfg.ci_pipeline_path if use_ci_model or not cfg.pipeline_path.exists() else cfg.pipeline_path
    sample = pd.read_csv(ARTIFACTS_DIR / "ci_sample.csv").head(5)
    with mlflow.start_run(run_name=f"best-{cfg.model_key}-{cfg.model_name}") as run:
        mlflow.set_tags({"model_key": cfg.model_key, "model_family": cfg.model_name, "is_best": "True",
                         "selected_model": "True", "objective": "f1", "pipeline_file": pkl.name})
        mlflow.log_params(_flat_params(cfg.hyperparameters))
        mlflow.log_params({"decision_threshold": round(cfg.decision_threshold, 6),
                           "imbalance_strategy": cfg.imbalance.get("strategy"),
                           "sampler": cfg.imbalance.get("sampler") or "none"})
        mlflow.log_metrics({f"test_{k}": float(cfg.test_metrics[k]) for k in METRIC_KEYS})
        mlflow.log_metrics({f"val_{k}": float(cfg.validation_metrics[k]) for k in METRIC_KEYS})
        for fig in sorted((ARTIFACTS_DIR / "figures").glob("*.png")):
            mlflow.log_artifact(str(fig), "figures")  # incl. feature importance, SHAP, ROC/PR, CM
        for f in ["best_model.json", "feature_spec.json", "model_comparison.csv", "cleaning_log.json",
                  "eda_summary.json", "imbalance_comparison.csv", "reload_check.json", "env_versions.json"]:
            if (ARTIFACTS_DIR / f).exists():
                mlflow.log_artifact(str(ARTIFACTS_DIR / f), "handoff")
        info = log_pyfunc_model(pkl, ARTIFACTS_DIR / "best_model.json", sample,
                                registered_model_name=MODEL_NAME if register else None)
        print(f"logged best model: run {run.info.run_id}  model URI {info.model_uri}")
        return run.info.run_id


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", action="store_true", help="log to ./mlruns instead of Azure")
    ap.add_argument("--register", action="store_true", help=f"register the best model as '{MODEL_NAME}'")
    ap.add_argument("--use-ci-model", action="store_true", help="log the small CI .pkl (e.g. from GitHub Actions)")
    ap.add_argument("--skip-comparison", action="store_true", help="only log the best-model run")
    a = ap.parse_args()

    uri = set_mlflow_tracking(local=a.local)
    print(f"MLflow tracking URI: {uri}\nExperiment: {EXPERIMENT_NAME}")
    cfg = load_best_model_config()
    with tempfile.TemporaryDirectory() as tmp:
        if not a.skip_comparison:
            summary = json.loads((ARTIFACTS_DIR / "all_models_summary.json").read_text())
            curves_path = ARTIFACTS_DIR / "curves.json"
            curves = json.loads(curves_path.read_text()) if curves_path.exists() else {}
            log_model_runs(summary, curves, cfg.model_key, Path(tmp))
        run_id = log_best_run(cfg, a.use_ci_model, a.register)
    if a.register:
        from mlops.register_model import tag_latest_version
        tag_latest_version(run_id)


if __name__ == "__main__":
    main()
