"""List (and optionally register) the Azure resource providers managed online endpoints need.

    python -m mlops.check_providers              # show registration state
    python -m mlops.check_providers --register   # register the ones that are NotRegistered

Registering needs Contributor/Owner rights on the subscription. If it is refused, send the
list of NotRegistered providers to your subscription admin.
"""
from __future__ import annotations

import argparse
import time

import requests
from azure.identity import DefaultAzureCredential

from mlops.azure_config import workspace_settings

REQUIRED = [
    "Microsoft.MachineLearningServices",
    "Microsoft.PolicyInsights",
    "Microsoft.Cdn",
    "Microsoft.ContainerRegistry",
    "Microsoft.Storage",
    "Microsoft.KeyVault",
    "Microsoft.Insights",
    "Microsoft.OperationalInsights",
    "Microsoft.Network",
]
API = "https://management.azure.com/subscriptions/{sub}/providers/{ns}{action}?api-version=2021-04-01"


def _headers() -> dict:
    token = DefaultAzureCredential().get_token("https://management.azure.com/.default").token
    return {"Authorization": f"Bearer {token}"}


def states(sub: str, headers: dict) -> dict:
    out = {}
    for ns in REQUIRED:
        r = requests.get(API.format(sub=sub, ns=ns, action=""), headers=headers, timeout=30)
        out[ns] = r.json().get("registrationState", f"HTTP {r.status_code}") if r.ok else f"HTTP {r.status_code}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--register", action="store_true")
    a = ap.parse_args()
    sub = workspace_settings()["subscription_id"]
    headers = _headers()
    current = states(sub, headers)
    for ns, st in current.items():
        print(f"{ns:<36} {st}")
    if not a.register:
        return
    todo = [ns for ns, st in current.items() if st == "NotRegistered"]
    if not todo:
        print("\nAll required providers are registered.")
        return
    for ns in todo:
        r = requests.post(API.format(sub=sub, ns=ns, action="/register"), headers=headers, timeout=30)
        if r.ok:
            msg = "requested"
        else:
            err = r.json().get("error", {}).get("message", r.text)[:200]
            msg = f"FAILED HTTP {r.status_code}: {err}"
        print(f"register {ns:<27} {msg}")
    print("\nWaiting for registration (up to ~3 min)...")
    for _ in range(18):
        time.sleep(10)
        pending = {ns: st for ns, st in states(sub, headers).items() if ns in todo and st != "Registered"}
        if not pending:
            print("All requested providers are now Registered.")
            return
    print("Still registering:", pending, "- re-run this script in a few minutes to check.")


if __name__ == "__main__":
    main()
