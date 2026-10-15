# Monitoring, retraining and safe rollout (Section 3, Task 6)

## 1. What is monitored

| Signal | Tool | Threshold / action |
|---|---|---|
| Requests, latency (p50/p95), HTTP 4xx/5xx, container logs | **Application Insights** (enabled on every deployment via `app_insights_enabled=True`) + endpoint metrics in Azure Monitor | Alert rule: p95 latency > 500 ms or 5xx rate > 1% for 5 min |
| Input data drift (PSI + KS per input feature, hour-of-day, balance discrepancies) | `monitoring/drift_check.py` against `artifacts/reference_sample.csv` (training period, true distribution); optionally Azure ML *model monitoring* (data drift signal) on collected inference data | PSI ≥ 0.10 warn, **PSI ≥ 0.25 alert** |
| Prediction drift | Share of transactions flagged as fraud per day (App Insights custom query on the API logs) | > 2× the validation alert rate |
| Model performance | F1 / precision / recall at the production threshold once labels arrive (chargebacks, investigations — typically weeks later) | **F1 drop ≥ 0.10** vs. `best_model.json` test F1 |

Example App Insights (KQL) query for endpoint latency and errors:

```kusto
requests
| where timestamp > ago(1d)
| summarize calls=count(), p95_ms=percentile(duration, 95), failures=countif(success == false) by bin(timestamp, 1h)
```

## 2. Retraining triggers

1. **Scheduled**: monthly retrain on the latest labelled window (`python -m src.train --mlflow`, or the same command as an
   Azure ML command job on `cpu-cluster` with an Azure ML schedule).
2. **Drift-based**: the weekly `drift_check` job exits with code 1 when any feature PSI ≥ 0.25 or F1 dropped ≥ 0.10;
   that starts the retraining job / `CD - Azure ML` workflow.
3. **Guardrail**: a retrained model is only registered as *Staging* if its validation F1 is ≥ the current Production F1
   (minus 0.01 tolerance); otherwise the run is kept for analysis and nothing is deployed.

Because hyperparameters were tuned in Colab, routine retraining refits the same configuration (fast). A full re-tuning
(the Colab notebook) is triggered only if F1 keeps dropping after a refit or the feature set changes.

## 3. Blue/green rollout

```
blue  (current Production, 100%)          green (new Staging version)
            │                                        │
1. deploy green with 0–10% traffic   python -m mlops.deploy_endpoint --slot green --version N --traffic 10
2. smoke test green directly         python -m mlops.test_endpoint --slot green
3. watch App Insights for 24 h (latency, errors, fraud-alert rate vs blue)
4. promote                           python -m mlops.deploy_endpoint --promote green
                                     python -m mlops.register_model --promote N
5. keep blue idle as rollback for a few days, then delete it
                                     python -m mlops.deploy_endpoint --delete-slot blue
```

**Rollback**: `python -m mlops.deploy_endpoint --promote blue` instantly returns all traffic to the previous version.
The next release goes to the now-empty `blue` slot, so the two slots alternate.

## 4. Cost control

Managed online endpoints bill per instance-hour even when idle. Take the screenshots, then run
`python -m mlops.deploy_endpoint --delete-endpoint`. Compute clusters scale to zero; stop the compute instance after use.
