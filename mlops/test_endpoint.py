"""Section 3, Task 5: call the managed online endpoint with its REST key.

    python -m mlops.test_endpoint                    # uses AZURE_ENDPOINT_URL / AZURE_ENDPOINT_KEY
    python -m mlops.test_endpoint --from-azure       # fetches URI + key with the Azure SDK
    python -m mlops.test_endpoint --slot green       # target one deployment directly

Sends artifacts/sample_requests.json (or ci_sample.csv rows), prints predictions,
agreement with the expected labels, and latency.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import pandas as pd

from api import azure_client
from src.config import ARTIFACTS_DIR
from src.features import INPUT_FEATURES


def load_requests(n: int) -> tuple[list[dict], list[int | None]]:
    path = ARTIFACTS_DIR / "sample_requests.json"
    if path.exists():
        items = json.loads(path.read_text())[:n]
        return [i["request"] for i in items], [i.get("expected_prediction") for i in items]
    df = pd.read_csv(ARTIFACTS_DIR / "ci_sample.csv").head(n)
    return df[INPUT_FEATURES].to_dict("records"), [None] * len(df)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-azure", action="store_true")
    ap.add_argument("--slot", default=None)
    ap.add_argument("-n", type=int, default=6)
    ap.add_argument("--repeat", type=int, default=5, help="calls per request for latency stats")
    a = ap.parse_args()

    if a.from_azure or not azure_client.is_configured():
        from mlops.azure_config import endpoint_name, ml_client
        ep = ml_client().online_endpoints.get(endpoint_name())
        os.environ["AZURE_ENDPOINT_URL"] = ep.scoring_uri
        os.environ["AZURE_ENDPOINT_KEY"] = ml_client().online_endpoints.get_keys(endpoint_name()).primary_key
    if a.slot:
        os.environ["AZURE_DEPLOYMENT"] = a.slot

    records, expected = load_requests(a.n)
    print(f"POST {os.environ['AZURE_ENDPOINT_URL']}  (deployment={os.getenv('AZURE_DEPLOYMENT', 'traffic split')})")
    print("payload example:", json.dumps(azure_client.build_payload(records[:1]))[:300], "...")
    preds = azure_client.score(records)
    agree = 0
    for rec, p, exp in zip(records, preds, expected, strict=True):
        flag = "" if exp is None else ("OK" if p["is_fraud"] == exp else f"MISMATCH (expected {exp})")
        agree += int(exp is not None and p["is_fraud"] == exp)
        print(f"  {rec['type']:<9} amount={rec['amount']:>14,.2f} -> prob={p['fraud_probability']:.4f} "
              f"is_fraud={p['is_fraud']} {flag}")
    if any(e is not None for e in expected):
        print(f"Agreement with Colab model: {agree}/{sum(e is not None for e in expected)}")

    lat = []
    for _ in range(a.repeat):
        t0 = time.perf_counter()
        azure_client.score(records[:1])
        lat.append((time.perf_counter() - t0) * 1000)
    print(f"Latency over {a.repeat} single-row calls: median {statistics.median(lat):.0f} ms, max {max(lat):.0f} ms")


if __name__ == "__main__":
    main()
