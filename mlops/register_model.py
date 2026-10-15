"""Section 3, Task 3: compare runs, register the best model, manage Staging/Production.

Azure ML's MLflow registry supports version tags reliably (aliases are tried
as well). Convention used here: tag `stage` = Staging | Production | Archived.

    python -m mlops.register_model --compare                 # rank runs by test F1
    python -m mlops.register_model --register-best           # register top run -> Staging
    python -m mlops.register_model --promote 2               # version 2 -> Production, others Archived
    python -m mlops.register_model --list
"""
from __future__ import annotations

import argparse

import mlflow
from mlflow.tracking import MlflowClient

from mlops.azure_config import EXPERIMENT_NAME, MODEL_NAME, set_mlflow_tracking


def _set_stage(client: MlflowClient, version: str, stage: str) -> None:
    client.set_model_version_tag(MODEL_NAME, version, "stage", stage)
    try:  # aliases are optional on Azure ML
        if stage in ("Staging", "Production"):
            client.set_registered_model_alias(MODEL_NAME, stage.lower(), version)
    except Exception:
        pass


def compare_runs(top: int = 10):
    """Rank runs by test F1 (sorted locally: Azure ML's MLflow server is strict about order_by)."""
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise SystemExit(f"Experiment '{EXPERIMENT_NAME}' not found; run mlops.track_experiments first.")
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], max_results=200)
    if runs.empty:
        print("No runs found.")
        return runs
    if "metrics.test_f1" in runs.columns:
        runs = runs.sort_values("metrics.test_f1", ascending=False)
    cols = [c for c in ["tags.mlflow.runName", "metrics.test_f1", "metrics.test_precision", "metrics.test_recall",
                        "metrics.test_pr_auc", "metrics.test_roc_auc", "metrics.latency_ms_single_row",
                        "tags.is_best"] if c in runs.columns]
    import pandas as pd

    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 4):
        print(runs[cols].head(top).rename(columns=lambda c: c.split(".", 1)[-1]).to_string(index=False))
    return runs


def tag_latest_version(run_id: str | None = None) -> str:
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if run_id:
        versions = [v for v in versions if v.run_id == run_id] or versions
    v = max(versions, key=lambda x: int(x.version))
    _set_stage(client, v.version, "Staging")
    client.set_model_version_tag(MODEL_NAME, v.version, "objective", "f1")
    print(f"{MODEL_NAME} v{v.version} tagged stage=Staging")
    return v.version


def register_best() -> str:
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], filter_string="tags.selected_model = 'True'")
    if runs.empty:
        raise SystemExit("No run with a logged model found; run mlops.track_experiments first.")
    run_id = runs.iloc[0]["run_id"]
    mv = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    MlflowClient().set_model_version_tag(MODEL_NAME, mv.version, "source_run", run_id)
    return tag_latest_version(run_id)


def promote(version: str) -> None:
    client = MlflowClient()
    for v in client.search_model_versions(f"name='{MODEL_NAME}'"):
        if v.version == str(version):
            _set_stage(client, v.version, "Production")
        elif v.tags.get("stage") == "Production":
            _set_stage(client, v.version, "Archived")
    print(f"{MODEL_NAME} v{version} -> Production")


def list_versions() -> None:
    for v in sorted(MlflowClient().search_model_versions(f"name='{MODEL_NAME}'"), key=lambda x: int(x.version)):
        print(f"v{v.version}  stage={v.tags.get('stage', '-')}  run={v.run_id}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--register-best", action="store_true")
    ap.add_argument("--promote", metavar="VERSION")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    set_mlflow_tracking(local=a.local)
    if a.compare:
        compare_runs()
    if a.register_best:
        register_best()
    if a.promote:
        promote(a.promote)
    if a.list or not any([a.compare, a.register_best, a.promote]):
        list_versions()


if __name__ == "__main__":
    main()
