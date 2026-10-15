"""Section 3, Task 6: data-drift and performance check -> retraining trigger.

Compares a batch of recent transactions with the training reference sample
(artifacts/reference_sample.csv) using the Population Stability Index (PSI)
for every model input and a KS test for numeric inputs. If labels are present
(`Is_Fraud`, available after chargebacks/investigations), F1 / precision /
recall at the production threshold are compared with the Colab test metrics.

    python -m monitoring.drift_check --current data/processed/last_week.csv
    python -m monitoring.drift_check --simulate       # demo: late time window vs reference

Exit code 1 = retraining recommended (usable from a scheduled job / GitHub Action).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config import ARTIFACTS_DIR, load_best_model_config
from src.features import RAW_CATEGORICAL, RAW_NUMERIC, TARGET, FraudFeatureEngineer

PSI_WARN, PSI_ALERT = 0.10, 0.25          # common industry thresholds
F1_DROP_ALERT = 0.10                      # absolute F1 drop vs. Colab test F1


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """PSI on reference quantile bins (robust to heavy tails)."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.array([-np.inf, np.median(ref), np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    cats = sorted(set(ref.astype(str)) | set(cur.astype(str)))
    r = ref.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=1e-6)
    c = cur.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=1e-6)
    return float(np.sum((c - r) * np.log(c / r)))


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame) -> dict:
    feats = {}
    eng_ref = FraudFeatureEngineer().transform(ref)
    eng_cur = FraudFeatureEngineer().transform(cur)
    # `step` is absolute time and always shifts -> monitor hour-of-day instead
    for col in [c for c in RAW_NUMERIC if c != "step"] + ["error_balance_orig", "error_balance_dest", "hour"]:
        a = (ref if col in ref else eng_ref)[col].to_numpy(dtype=float)
        b = (cur if col in cur else eng_cur)[col].to_numpy(dtype=float)
        p = psi(a, b)
        ks = ks_2samp(a, b)
        feats[col] = {"psi": round(p, 4), "ks_stat": round(float(ks.statistic), 4),
                      "ks_pvalue": float(ks.pvalue),
                      "status": "alert" if p >= PSI_ALERT else "warn" if p >= PSI_WARN else "ok"}
    for col in RAW_CATEGORICAL:
        p = psi_categorical(ref[col], cur[col])
        feats[col] = {"psi": round(p, 4),
                      "status": "alert" if p >= PSI_ALERT else "warn" if p >= PSI_WARN else "ok"}
    return feats


def performance_report(cur: pd.DataFrame) -> dict | None:
    if TARGET not in cur or cur[TARGET].sum() == 0:
        return None
    import joblib

    from src.evaluate import metrics_at

    cfg = load_best_model_config()
    path = cfg.pipeline_path if cfg.pipeline_path.exists() else cfg.ci_pipeline_path
    pipe = joblib.load(path)
    m = metrics_at(cur[TARGET], pipe.predict_proba(cur[RAW_NUMERIC + RAW_CATEGORICAL])[:, 1],
                   cfg.decision_threshold)
    base = cfg.test_metrics["f1"]
    m["baseline_f1"] = base
    m["f1_drop"] = round(base - m["f1"], 4)
    m["status"] = "alert" if m["f1_drop"] >= F1_DROP_ALERT else "ok"
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", default=str(ARTIFACTS_DIR / "reference_sample.csv"))
    ap.add_argument("--current", help="CSV of recent transactions (raw input columns, optional Is_Fraud)")
    ap.add_argument("--simulate", action="store_true", help="use ci_sample.csv (test period) as 'current'")
    ap.add_argument("--out", default=str(ARTIFACTS_DIR / "monitoring" / "drift_report.json"))
    a = ap.parse_args()

    ref = pd.read_csv(a.reference)
    cur_path = ARTIFACTS_DIR / "ci_sample.csv" if a.simulate or not a.current else Path(a.current)
    cur = pd.read_csv(cur_path)
    feats = drift_report(ref, cur)
    perf = performance_report(cur)
    alerts = [f for f, v in feats.items() if v["status"] == "alert"]
    retrain = bool(alerts) or (perf is not None and perf["status"] == "alert")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "reference": str(a.reference), "current": str(cur_path), "rows_current": len(cur),
              "features": feats, "performance": perf, "drifted_features": alerts,
              "retrain_recommended": retrain,
              "thresholds": {"psi_warn": PSI_WARN, "psi_alert": PSI_ALERT, "f1_drop_alert": F1_DROP_ALERT}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))

    print(pd.DataFrame(feats).T.to_string())
    if perf:
        print(f"\nF1 now {perf['f1']:.4f} vs baseline {perf['baseline_f1']:.4f} (drop {perf['f1_drop']:+.4f})")
    print(f"\nRetraining recommended: {retrain}  (report: {a.out})")
    sys.exit(1 if retrain else 0)


if __name__ == "__main__":
    main()
