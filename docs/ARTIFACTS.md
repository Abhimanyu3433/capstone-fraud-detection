# Colab → VS Code hand-off (`artifacts/`)

Part 1 (Colab notebook `notebooks/01_colab_model_selection.ipynb`) writes this folder to
`MyDrive/fraud-detection/artifacts/`. Part 2 (this VS Code project) reads it through `src/config.py`.
Copy the folder into the project root unchanged (or unzip `artifacts_bundle.zip`).

| File | Contents | Used by (Part 2) |
|---|---|---|
| `best_model.json` | Best model key/class, objective (F1), tuned hyperparameters, imbalance strategy, **decision threshold** (F1-optimal), CV / validation / test metrics, latency | training script, MLflow params, API threshold, report |
| `feature_spec.json` | API input fields, dtypes, observed ranges, allowed `type` values, engineered features, dropped columns | Pydantic schemas, tests |
| `env_versions.json` | Library versions used in Colab | pin `requirements.txt` so the `.pkl` loads |
| `models/fraud_pipeline.pkl` | Final fitted pipeline (preprocessing + model), compressed. Git-ignored | API locally, MLflow model logging, Azure endpoint |
| `models/fraud_pipeline_ci.pkl` | Same architecture and params, small ensemble, fitted on 20k rows. Committed | pytest + GitHub Actions |
| `model_comparison.csv` | One row per model: CV/val PR-AUC, threshold, test Accuracy/Precision/Recall/F1/ROC-AUC/PR-AUC, confusion counts, latency | report, MLflow comparison runs |
| `all_models_summary.json` | Full search + evaluation details for every model | MLflow: one run per model |
| `cv_results/<model>.csv` | Raw `RandomizedSearchCV.cv_results_` | report appendix |
| `imbalance_comparison.csv` | none / class_weight / under / over / SMOTE comparison | report |
| `cleaning_log.json` | Missing values, duplicates, outlier decisions, dropped columns and reasons | report |
| `eda_summary.json` | Key EDA numbers (class balance, fraud by type/hour, correlations) | report |
| `feature_importance.csv`, `permutation_importance.csv`, `shap_importance.csv` | Interpretation of the best model | report |
| `curves.json` | Down-sampled ROC / PR curve points per model (test set) | per-run ROC/PR plots in MLflow |
| `reference_sample.csv` | 20,000 training-period rows (true distribution) | drift monitoring (`monitoring/drift_check.py`) |
| `ci_sample.csv` | 2,000 labelled test rows (incl. ~100 frauds), raw input columns | tests, Azure endpoint smoke test |
| `sample_requests.json` | 6 ready-made `/predict` request bodies with expected outputs | tests, Swagger/Postman, screenshots |
| `reload_check.json` | Proof that the reloaded `.pkl` reproduces predictions | report (Task 11) |
| `run_manifest.json` | Timestamp, budget, split boundaries, timings, file list | traceability |
| `figures/*.png` | 13 EDA / evaluation / interpretation figures | report |
| `checkpoints/` | Search checkpoints for resuming on Colab — **do not copy** | — |

## Key design decisions recorded in the artifacts

- **Time-based split** on `step`: train ≈ first 70% of rows, validation ≈ next 15%, test ≈ last 15%.
- **Development sample** for tuning: all training frauds + legit sampled at `neg_pos_ratio` per fraud.
  Validation and test keep the true class distribution.
- **Resampling inside the CV pipeline** (imblearn `Pipeline`), never before the split.
- **Objective: F1** (fraud class). Imbalance strategy chosen by CV F1, searches refit on F1,
  **threshold** = F1-optimal on validation, **model selection** = highest validation F1 (ties → CV F1); test touched once.
- **Final model: Random Forest** (chosen for deployment via `FINAL_MODEL`); the criterion-based pick is stored as `criterion_pick`.
  The minimum-cost threshold (`COST_FN : COST_FP = 25 : 1`) is stored as `threshold_min_cost` for the report.
- Raw probabilities are inflated by the negative downsampling, so always use `decision_threshold`, not 0.5.
