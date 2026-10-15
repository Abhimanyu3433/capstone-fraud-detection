"""Section 3, Task 1: Azure ML workspace + compute.

The workspace (with its storage account, key vault, Application Insights and,
on first use, container registry) can be created in the portal or here:

    python -m mlops.azure_setup --create-workspace --location centralindia
    python -m mlops.azure_setup --cluster cpu-cluster --instance ci-<yourname>

Compute clusters scale to 0 nodes when idle (no cost); stop the compute
instance when you are not using it.
"""
from __future__ import annotations

import argparse

from mlops.azure_config import ml_client, workspace_settings


def create_workspace(location: str) -> None:
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import Workspace
    from azure.identity import DefaultAzureCredential

    s = workspace_settings()
    client = MLClient(DefaultAzureCredential(), s["subscription_id"], s["resource_group"])
    ws = Workspace(name=s["workspace_name"], location=location,
                   display_name="Fraud detection capstone",
                   description="Capstone Project 1 - online fraud detection (MLflow tracking, registry, endpoints)")
    ws = client.workspaces.begin_create(ws).result()
    print(f"Workspace ready: {ws.name} ({ws.location})")
    print(f"  storage: {ws.storage_account}\n  key vault: {ws.key_vault}\n  app insights: {ws.application_insights}")


def ensure_cluster(name: str, size: str, max_nodes: int) -> None:
    from azure.ai.ml.entities import AmlCompute

    try:
        c = ml_client().compute.get(name)
        print(f"Cluster exists: {c.name} ({c.size})")
    except Exception:
        c = AmlCompute(name=name, size=size, min_instances=0, max_instances=max_nodes,
                       idle_time_before_scale_down=300)
        ml_client().compute.begin_create_or_update(c).result()
        print(f"Cluster created: {name} ({size}, 0-{max_nodes} nodes)")


def ensure_instance(name: str, size: str) -> None:
    from azure.ai.ml.entities import ComputeInstance

    try:
        c = ml_client().compute.get(name)
        print(f"Compute instance exists: {c.name} ({c.size}, state={c.state})")
    except Exception:
        ci = ComputeInstance(name=name, size=size, idle_time_before_shutdown_minutes=60)
        ml_client().compute.begin_create_or_update(ci).result()
        print(f"Compute instance created: {name} ({size}); auto-shutdown after 60 idle minutes")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--create-workspace", action="store_true")
    ap.add_argument("--location", default="centralindia")
    ap.add_argument("--cluster", default="cpu-cluster")
    ap.add_argument("--cluster-size", default="Standard_DS3_v2")
    ap.add_argument("--max-nodes", type=int, default=1)
    ap.add_argument("--instance", default=None, help="compute instance name (globally unique per region)")
    ap.add_argument("--instance-size", default="Standard_DS3_v2")
    a = ap.parse_args()
    if a.create_workspace:
        create_workspace(a.location)
    if a.cluster:
        ensure_cluster(a.cluster, a.cluster_size, a.max_nodes)
    if a.instance:
        ensure_instance(a.instance, a.instance_size)


if __name__ == "__main__":
    main()
