# Online Fraud Transaction Detection — End-to-End ML, FastAPI CI and Azure MLOps

Capstone Project 1, Option A. Author: Abhimanyu Kumar Singh.

A binary classifier (`Is_Fraud`) trained with Scikit-learn, served via FastAPI with GitHub Actions CI,
and tracked, registered and deployed on Azure Machine Learning with MLflow. **Optimisation target: F1** (fraud class).

## Workflow

| Part | Where | What |
|---|---|---|
| **1. Model selection** | Google Colab (`notebooks/01_colab_model_selection.ipynb`), data in Google Drive | Loading, cleaning, EDA, feature engineering, imbalance study, tuning of 5 models, evaluation, F1 threshold, interpretation, `.pkl` export |
| **2. Application** | VS Code (this repo) | Reads `artifacts/` → FastAPI + pytest + GitHub Actions CI → MLflow tracking/registry on Azure ML → managed online endpoint → monitoring |

The hand-off format is documented in [`docs/ARTIFACTS.md`](docs/ARTIFACTS.md). `src/features.py` is shared by both
parts, so the pickled pipeline loads in Colab, locally, in CI and on Azure.

```
capstone-fraud/
├── artifacts/            # Colab hand-off: best_model.json, feature_spec.json, .pkl, metrics, figures
├── notebooks/            # 01_colab_model_selection.ipynb (Part 1)
├── src/                  # features.py (shared), config.py, data.py, evaluate.py, train.py, pyfunc_model.py
├── api/                  # FastAPI app (main.py), schemas.py, model_loader.py, azure_client.py
├── tests/                # pytest: API validation, features, model artefact
├── mlops/                # Azure ML setup, MLflow tracking, registry, endpoint deploy/test
├── monitoring/           # drift_check.py (PSI/KS + F1 drop -> retrain trigger)
├── docs/                 # ARTIFACTS.md, MONITORING.md
├── postman/              # Postman collection for the API
├── .github/workflows/    # ci.yml (lint + tests), cd-azure.yml (optional CD)
├── requirements.txt      # full environment;  requirements-ci.txt: CI / API only
└── pyproject.toml        # ruff + pytest config
```

---

## Step 1 — Colab: model selection (Section 1)

1. Google Drive: dataset at `MyDrive/fraud-detection/data/onlinefraud.csv`; upload `src/__init__.py` and
   `src/features.py` from this repo to `MyDrive/fraud-detection/src/`.
2. Upload `notebooks/01_colab_model_selection.ipynb` to Colab → Run all (`BUDGET = "quick"`, ≈20–30 min,
   checkpointed — re-run after a disconnect).
3. Download `MyDrive/fraud-detection/artifacts_bundle.zip` and unzip it into this repo's `artifacts/` folder.

## Step 2 — VS Code: environment

```powershell
cd $HOME\Documents\capstone-fraud
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python setup/install_config.py  # puts the CI workflows in .github/workflows and VS Code settings in .vscode
python -m src.config            # validates the hand-off and compares library versions with Colab
python -m src.config --pin      # if versions differ: pin them, then pip install -r requirements.txt again
```

## Step 3 — FastAPI service and tests (Section 2)

```powershell
uvicorn api.main:app --reload    # or VS Code: Run and Debug -> "FastAPI (uvicorn --reload)"
```

- Swagger UI: http://127.0.0.1:8000/docs
- Endpoints: `GET /health`, `GET /model/info`, `POST /predict`, `POST /predict/batch`, `POST /predict/azure`
- Postman: import `postman/fraud_api.postman_collection.json` (valid + invalid requests with test scripts)

```powershell
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" `
  -d '{"step":410,"type":"TRANSFER","amount":181000,"oldbalanceOrg":181000,"newbalanceOrig":0,"oldbalanceDest":0,"newbalanceDest":0}'

ruff check .
pytest -v
```

Tests cover valid input, missing fields, wrong data types, out-of-range values, unseen categories, unexpected
fields, batch limits, the Azure client path (mocked), feature engineering and `.pkl` reload equality.

## Step 4 — Git and CI (Section 2)

```powershell
git init; git branch -M main
git add .; git commit -m "Initial project structure and Colab hand-off"
git remote add origin https://github.com/<you>/capstone-fraud.git
git push -u origin main

git checkout -b feature/fastapi-service      # one branch per feature, then open a PR into main
git push -u origin feature/fastapi-service
```

`.github/workflows/ci.yml` runs on every push and pull request: install `requirements-ci.txt` → `ruff check .` →
`pytest`. Any failing test fails the build. The small `artifacts/models/fraud_pipeline_ci.pkl` is committed for CI;
the full `fraud_pipeline.pkl` is git-ignored.

## Step 5 — Azure ML + MLflow (Section 3)

Prerequisites: Azure CLI (`az login`), and either `config.json` from the workspace in the project root or the
variables in `.env.example`.

```powershell
# 1. workspace + compute (or create the workspace in the portal)
python -m mlops.azure_setup --create-workspace --location centralindia
python -m mlops.azure_setup --cluster cpu-cluster --instance ci-abhimanyu

# 2. experiment tracking: one run per model + best run with model, plots, figures
python -m mlops.track_experiments --local      # optional dry run -> mlflow ui
python -m mlops.track_experiments --register   # Azure: logs runs and registers the best model (stage=Staging)

# 3. compare runs and promote
python -m mlops.register_model --compare
python -m mlops.register_model --promote 1

# 4. managed online endpoint (App Insights enabled) + blue/green
python -m mlops.deploy_endpoint --slot blue --traffic 100
python -m mlops.deploy_endpoint --show

# 5. endpoint testing: REST key from Python, and FastAPI as a client
python -m mlops.test_endpoint --from-azure
$env:AZURE_ENDPOINT_URL="<scoring uri>"; $env:AZURE_ENDPOINT_KEY="<key>"
uvicorn api.main:app --reload                  # then POST /predict/azure in Swagger

# 6. monitoring / retraining
python -m monitoring.drift_check --simulate
python -m src.train --max-rows 1500000 --mlflow  # time-bounded refit with the Colab hyperparameters

# clean up (endpoints bill per hour)
python -m mlops.deploy_endpoint --delete-endpoint
```

Monitoring, retraining triggers and the blue/green procedure are described in [`docs/MONITORING.md`](docs/MONITORING.md).
Optional CD: `.github/workflows/cd-azure.yml` (manual trigger; needs a service principal in repo secrets).

## Screenshot checklist (report)

- [ ] Dataset preview and cleaning log (Colab sections 1–2)
- [ ] EDA figures (`artifacts/figures/01–07`)
- [ ] Model comparison table, CV results (Colab sections 5–7)
- [ ] Confusion matrices and ROC/PR curves (`09`, `10`, `11`)
- [ ] Saved `.pkl` + reload check (Colab section 9)
- [ ] Swagger UI `/docs` and Postman runs (valid + 422 cases)
- [ ] GitHub Actions CI passing
- [ ] MLflow experiment runs comparison + Model Registry (versions with stage tags)
- [ ] Azure endpoint (deployments, traffic), test call output, Application Insights
