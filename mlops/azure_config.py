"""Azure ML connection settings shared by all mlops scripts.

The workspace is found automatically, in this order:
  1. env vars AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_ML_WORKSPACE
  2. on an Azure ML compute instance: the AZUREML_ARM_* variables it sets (no login needed)
  3. the workspace `config.json` (Studio -> workspace -> Download config file) in the project root
Authentication uses DefaultAzureCredential: the compute instance's identity, or `az login` locally.
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = os.getenv("AZURE_MODEL_NAME", "fraud-detection-model")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "fraud-detection")
INSTANCE_TYPE = os.getenv("AZURE_INSTANCE_TYPE", "Standard_DS3_v2")


def workspace_settings() -> dict:
    env = {k: os.getenv(v) for k, v in [("subscription_id", "AZURE_SUBSCRIPTION_ID"),
                                        ("resource_group", "AZURE_RESOURCE_GROUP"),
                                        ("workspace_name", "AZURE_ML_WORKSPACE")]}
    if all(env.values()):
        return env
    ci = {k: os.getenv(v) for k, v in [("subscription_id", "AZUREML_ARM_SUBSCRIPTION"),
                                       ("resource_group", "AZUREML_ARM_RESOURCEGROUP"),
                                       ("workspace_name", "AZUREML_ARM_WORKSPACE_NAME")]}
    if all(ci.values()):  # running on an Azure ML compute instance
        return ci
    cfg = PROJECT_ROOT / "config.json"
    if cfg.exists():
        d = json.loads(cfg.read_text())
        return {"subscription_id": d["subscription_id"], "resource_group": d["resource_group"],
                "workspace_name": d["workspace_name"]}
    raise RuntimeError("Set AZURE_SUBSCRIPTION_ID / AZURE_RESOURCE_GROUP / AZURE_ML_WORKSPACE "
                       "or put the workspace config.json in the project root.")


def endpoint_name() -> str:
    """Endpoint names must be unique per Azure region, so the default adds a short workspace hash."""
    if os.getenv("AZURE_ENDPOINT_NAME"):
        return os.environ["AZURE_ENDPOINT_NAME"]
    s = workspace_settings()
    suffix = hashlib.sha1(f"{s['subscription_id']}/{s['workspace_name']}".encode()).hexdigest()[:6]
    return f"fraud-ep-{suffix}"


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
