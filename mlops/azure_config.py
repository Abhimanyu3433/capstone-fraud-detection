"""Azure ML connection settings shared by all mlops scripts.

Provide the workspace either with environment variables
    AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_ML_WORKSPACE
or by placing the workspace `config.json` (Azure ML studio -> workspace -> Download config file)
in the project root (it is git-ignored). Authenticate first with `az login`.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = os.getenv("AZURE_MODEL_NAME", "fraud-detection-model")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "fraud-detection")
ENDPOINT_NAME = os.getenv("AZURE_ENDPOINT_NAME", "fraud-detect-ep")
INSTANCE_TYPE = os.getenv("AZURE_INSTANCE_TYPE", "Standard_DS3_v2")


def workspace_settings() -> dict:
    env = {k: os.getenv(v) for k, v in [("subscription_id", "AZURE_SUBSCRIPTION_ID"),
                                        ("resource_group", "AZURE_RESOURCE_GROUP"),
                                        ("workspace_name", "AZURE_ML_WORKSPACE")]}
    if all(env.values()):
        return env
    cfg = PROJECT_ROOT / "config.json"
    if cfg.exists():
        d = json.loads(cfg.read_text())
        return {"subscription_id": d["subscription_id"], "resource_group": d["resource_group"],
                "workspace_name": d["workspace_name"]}
    raise RuntimeError("Set AZURE_SUBSCRIPTION_ID / AZURE_RESOURCE_GROUP / AZURE_ML_WORKSPACE "
                       "or put the workspace config.json in the project root.")


@lru_cache
def ml_client():
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    return MLClient(DefaultAzureCredential(), **workspace_settings())


def set_mlflow_tracking(local: bool = False) -> str:
    """Point MLflow at the Azure ML workspace (or ./mlruns when local=True)."""
    import mlflow

    if local:
        uri = (PROJECT_ROOT / "mlruns").as_uri()
    else:
        ws = ml_client().workspaces.get(workspace_settings()["workspace_name"])
        uri = ws.mlflow_tracking_uri
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return uri
