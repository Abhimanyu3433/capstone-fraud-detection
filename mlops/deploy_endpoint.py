"""Section 3, Tasks 4-6: managed online endpoint with blue/green rollout.

    # first deployment ("blue", 100% traffic) of the Production (or latest) model version
    python -m mlops.deploy_endpoint --slot blue --traffic 100
    # new version on "green" with 10% canary traffic, then promote and remove blue
    python -m mlops.deploy_endpoint --slot green --version 3 --traffic 10
    python -m mlops.deploy_endpoint --promote green
    python -m mlops.deploy_endpoint --delete-slot blue
    # show scoring URI / keys / traffic, and delete everything when done (endpoints bill per hour!)
    python -m mlops.deploy_endpoint --show
    python -m mlops.deploy_endpoint --delete-endpoint

Application Insights logging (requests, latency, errors, stdout) is enabled on every deployment.
"""
from __future__ import annotations

import argparse

from mlops.azure_config import INSTANCE_TYPE, MODEL_NAME, endpoint_name, ml_client


def resolve_model_version(version: str | None) -> str:
    if version:
        return version
    models = list(ml_client().models.list(name=MODEL_NAME))
    prod = [m for m in models if (m.tags or {}).get("stage") == "Production"]
    chosen = max(prod or models, key=lambda m: int(m.version))
    print(f"Using {MODEL_NAME} v{chosen.version} (stage={(chosen.tags or {}).get('stage', '-')})")
    return chosen.version


def ensure_endpoint():
    from azure.ai.ml.entities import ManagedOnlineEndpoint

    try:
        return ml_client().online_endpoints.get(endpoint_name())
    except Exception:
        ep = ManagedOnlineEndpoint(name=endpoint_name(), auth_mode="key",
                                   description="Online fraud detection (Capstone Project 1)",
                                   tags={"project": "fraud-capstone"})
        print(f"Creating endpoint {endpoint_name()} ...")
        return ml_client().online_endpoints.begin_create_or_update(ep).result()


def deploy(slot: str, version: str | None, traffic: int, instance_type: str, instance_count: int) -> None:
    from azure.ai.ml.entities import ManagedOnlineDeployment

    ensure_endpoint()
    v = resolve_model_version(version)
    model = ml_client().models.get(name=MODEL_NAME, version=v)
    dep = ManagedOnlineDeployment(
        name=slot, endpoint_name=endpoint_name(), model=model,
        instance_type=instance_type, instance_count=instance_count,
        app_insights_enabled=True,  # usage, latency and error telemetry
        tags={"model_version": v},
    )
    print(f"Deploying {MODEL_NAME} v{v} to {endpoint_name()}/{slot} ({instance_count} x {instance_type}) ...")
    ml_client().online_deployments.begin_create_or_update(dep).result()
    set_traffic(slot, traffic)


def set_traffic(slot: str, pct: int) -> None:
    ep = ml_client().online_endpoints.get(endpoint_name())  # refresh after deployment
    current = dict(ep.traffic or {})
    others = [d for d in current if d != slot]
    if not others or pct >= 100:
        new = {slot: 100, **{d: 0 for d in others}}
    else:
        rest = 100 - pct
        share = {d: current.get(d, 0) for d in others}
        total = sum(share.values()) or len(others)
        new = {slot: pct}
        alloc = 0
        for i, d in enumerate(others):
            val = rest - alloc if i == len(others) - 1 else round(rest * (share[d] or 1) / total)
            new[d] = val
            alloc += val
    ep.traffic = new
    ml_client().online_endpoints.begin_create_or_update(ep).result()
    print(f"Traffic: {new}")


def delete_slot(slot: str) -> None:
    ep = ml_client().online_endpoints.get(endpoint_name())
    if (ep.traffic or {}).get(slot, 0) > 0:
        raise SystemExit(f"{slot} still receives traffic; promote the other slot first.")
    ml_client().online_deployments.begin_delete(name=slot, endpoint_name=endpoint_name()).result()
    print(f"Deleted deployment {slot}")


def show() -> None:
    ep = ml_client().online_endpoints.get(endpoint_name())
    keys = ml_client().online_endpoints.get_keys(endpoint_name())
    print(f"Endpoint : {ep.name}\nScoring  : {ep.scoring_uri}\nTraffic  : {ep.traffic}")
    print(f"Key      : {keys.primary_key[:6]}... (full key: set AZURE_ENDPOINT_KEY from the studio 'Consume' tab)")
    for d in ml_client().online_deployments.list(endpoint_name()):
        print(f"  - {d.name}: {d.instance_count} x {d.instance_type}, model={d.model}")


def logs(slot: str, lines: int = 200) -> None:
    """Container logs of a deployment (also works after a failed deployment)."""
    for container in ("storage-initializer", "inference-server"):
        print(f"===== {slot} / {container} (last {lines} lines) =====")
        try:
            print(ml_client().online_deployments.get_logs(
                name=slot, endpoint_name=endpoint_name(), lines=lines,
                container_type=None if container == "inference-server" else container))
        except Exception as e:
            print(f"(no logs: {e})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slot", choices=["blue", "green"])
    ap.add_argument("--version", help="model version (default: Production-tagged, else latest)")
    ap.add_argument("--traffic", type=int, default=100)
    ap.add_argument("--instance-type", default=INSTANCE_TYPE)
    ap.add_argument("--instance-count", type=int, default=1)
    ap.add_argument("--promote", choices=["blue", "green"], help="send 100% traffic to this slot")
    ap.add_argument("--delete-slot", choices=["blue", "green"])
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--logs", choices=["blue", "green"], help="print container logs of a deployment")
    ap.add_argument("--delete-endpoint", action="store_true")
    a = ap.parse_args()
    if a.slot:
        deploy(a.slot, a.version, a.traffic, a.instance_type, a.instance_count)
    if a.promote:
        set_traffic(a.promote, 100)
    if a.delete_slot:
        delete_slot(a.delete_slot)
    if a.show:
        show()
    if a.logs:
        logs(a.logs)
    if a.delete_endpoint:
        ml_client().online_endpoints.begin_delete(name=endpoint_name()).result()
        print(f"Deleted endpoint {endpoint_name()}")


if __name__ == "__main__":
    main()
