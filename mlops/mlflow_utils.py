"""MLflow helpers: logging the pyfunc model and retraining runs."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from mlops.azure_config import PROJECT_ROOT
from src.evaluate import (
    curves_from_proba,
    model_feature_importance,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_pr,
)
from src.features import INPUT_FEATURES

PIN_PACKAGES = ["scikit-learn", "imbalanced-learn", "xgboost", "numpy", "pandas", "scipy", "joblib"]


def pip_requirements() -> list[str]:
    """Pinned serving requirements so the Azure image matches the training environment."""
    from importlib import metadata

    reqs = []
    for pkg in PIN_PACKAGES:
        try:
            reqs.append(f"{pkg}=={metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            pass
    import mlflow
    reqs.append(f"mlflow=={mlflow.__version__}")
    return reqs


def normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    """JSON-friendly dtypes for the MLflow signature: step=long, money=double, type=string.

    (int16/category from read_csv would make the endpoint reject normal JSON requests.)
    """
    df = df[INPUT_FEATURES].copy().reset_index(drop=True)
    df["step"] = df["step"].astype("int64")
    df["type"] = df["type"].astype(str)
    for c in INPUT_FEATURES:
        if c not in ("step", "type"):
            df[c] = df[c].astype("float64")
    return df


def log_pyfunc_model(pipeline_path: Path, config_path: Path, input_example: pd.DataFrame,
                     registered_model_name: str | None = None):
    """Log the pipeline as an MLflow pyfunc that returns probability + thresholded label."""
    import mlflow
    from mlflow.models import infer_signature

    from src.pyfunc_model import FraudDetectionModel

    input_example = normalize_input(input_example)
    wrapper = FraudDetectionModel()

    class _Ctx:  # build the signature from a real prediction
        artifacts = {"pipeline": str(pipeline_path), "config": str(config_path)}

    wrapper.load_context(_Ctx())
    signature = infer_signature(input_example, wrapper.predict(None, input_example))
    return mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=FraudDetectionModel(),
        artifacts={"pipeline": str(pipeline_path), "config": str(config_path)},
        code_paths=[str(PROJECT_ROOT / "src")],
        pip_requirements=pip_requirements(),
        signature=signature,
        input_example=input_example,
        registered_model_name=registered_model_name,
    )


def log_retrain_run(pipe, summary: dict, X_test, y_test, test_proba, experiment: str) -> str:
    """Log a src.train run (params, metrics, plots, model)."""
    import joblib
    import mlflow

    from mlops.azure_config import set_mlflow_tracking

    try:
        set_mlflow_tracking()
    except Exception as e:  # no Azure config -> local tracking
        print(f"Azure tracking unavailable ({e}); logging to ./mlruns")
        set_mlflow_tracking(local=True)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"retrain-{summary['model']}") as run, tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mlflow.set_tags({"stage": "retrain", "model_family": summary["model"]})
        mlflow.log_params({k: v for k, v in summary["hyperparameters"].items()})
        mlflow.log_params({"neg_pos_ratio": summary["neg_pos_ratio"], "rows_used": summary["rows_used"],
                           "decision_threshold": round(summary["decision_threshold"], 6)})
        m = summary["test_metrics"]
        mlflow.log_metrics({f"test_{k}": v for k, v in m.items() if k != "threshold"})
        mlflow.log_artifact(plot_confusion_matrix(m, "Confusion matrix (test)", tmp / "confusion_matrix.png"))
        c = curves_from_proba(y_test, test_proba)
        mlflow.log_artifact(plot_roc_pr(c["roc"]["fpr"], c["roc"]["tpr"], c["pr"]["precision"],
                                        c["pr"]["recall"], "Test curves", tmp / "roc_pr.png", c["prevalence"]))
        fi = model_feature_importance(pipe)
        if fi:
            mlflow.log_artifact(plot_feature_importance(*fi, "Feature importance", tmp / "feature_importance.png"))
        pkl = tmp / "fraud_pipeline.pkl"
        joblib.dump(pipe, pkl, compress=3)
        cfg = tmp / "best_model.json"
        cfg.write_text(json.dumps({"decision_threshold": summary["decision_threshold"]}))
        log_pyfunc_model(pkl, cfg, X_test[INPUT_FEATURES].head(3))
        return run.info.run_id
