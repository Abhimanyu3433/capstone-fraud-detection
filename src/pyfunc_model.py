"""MLflow pyfunc wrapper: returns the fraud probability AND the thresholded label.

A plain `mlflow.sklearn` model would only return `predict()` (labels at 0.5).
Our model needs the F1-tuned threshold from best_model.json, so the registered
model wraps the pipeline and applies it. Azure ML deploys this MLflow model to
a managed online endpoint without a scoring script.
"""
from __future__ import annotations

import json

import mlflow.pyfunc
import pandas as pd

from src.features import INPUT_FEATURES


class FraudDetectionModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import joblib

        self.pipeline = joblib.load(context.artifacts["pipeline"])
        with open(context.artifacts["config"]) as f:
            self.threshold = float(json.load(f)["decision_threshold"])

    def predict(self, context, model_input, params=None):
        df = pd.DataFrame(model_input)[INPUT_FEATURES]
        proba = self.pipeline.predict_proba(df)[:, 1]
        return pd.DataFrame({
            "fraud_probability": proba,
            "is_fraud": (proba >= self.threshold).astype(int),
        })
