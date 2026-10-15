"""Azure ML connection settings shared by all mlops scripts.

The workspace is found automatically, in this order:
  1. env vars AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_ML_WORKSPACE
  2. the workspace `config.json` in the project root (setup/azure_ci_setup.sh writes it)
  3. AZUREML_ARM_* variables (Azure ML jobs) or compute-instance metadata
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


def _clean(d: dict) -> dict | None:
    """Return d only if every value is a real setting (compute instances may export the text 'None')."""
    if all(v and str(v).strip().lower() not in ("none", "null", "") for v in d.values()):
        return {k: str(v).strip() for k, v in d.items()}
    return None


def workspace_settings() -> dict:
    env = _clean({k: os.getenv(v) for k, v in [("subscription_id", "AZURE_SUBSCRIPTION_ID"),
                                               ("resource_group", "AZURE_RESOURCE_GROUP"),
                                               ("workspace_name", "AZURE_ML_WORKSPACE")]})
    if env:
        return env
    for cfg in [PROJECT_ROOT / "config.json", PROJECT_ROOT / ".azureml" / "config.json",
                Path.home() / ".azureml" / "config.json", Path.home() / "cloudfiles" / "code" / "config.json"]:
        if cfg.exists():
            d = json.loads(cfg.read_text())
            found = _clean({"subscription_id": d.get("subscription_id"), "resource_group": d.get("resource_group"),
                            "workspace_name": d.get("workspace_name")})
            if found:
                return found
    job = _clean({k: os.getenv(v) for k, v in [("subscription_id", "AZUREML_ARM_SUBSCRIPTION"),
                                               ("resource_group", "AZUREML_ARM_RESOURCEGROUP"),
                                               ("workspace_name", "AZUREML_ARM_WORKSPACE_NAME")]})
    if job:  # running inside an Azure ML job
        return job
    nbvm = _compute_instance_settings()
    if nbvm:  # compute instance metadata
        return nbvm
    raise RuntimeError("Workspace not found. In Azure ML studio click your workspace name (top right) -> "
                       "'Download config file', and upload config.json into the project folder; or set "
                       "AZURE_SUBSCRIPTION_ID / AZURE_RESOURCE_GROUP / AZURE_ML_WORKSPACE.")


def _compute_instance_settings() -> dict | None:
    """Compute instances record their workspace in /mnt/azmnt/.nbvm (key=value lines) and CI_* env vars."""
    vals = {}
    nbvm = Path("/mnt/azmnt/.nbvm")
    if nbvm.exists():
        for line in nbvm.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                vals[k.strip().lower()] = v.strip().strip('"')
    sub = vals.get("subscription") or vals.get("subscription_id") or os.getenv("CI_SUBSCRIPTION")
    rg = vals.get("resourcegroup") or vals.get("resource_group") or os.getenv("CI_RESOURCE_GROUP")
    ws = vals.get("workspace") or vals.get("workspace_name") or os.getenv("CI_WORKSPACE")
    return _clean({"subscription_id": sub, "resource_group": rg, "workspace_name": ws})


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

    s = workspace_settings()
    return MLClient(DefaultAzureCredential(), subscription_id=s["subscription_id"],
                    resource_group_name=s["resource_group"], workspace_name=s["workspace_name"])


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
