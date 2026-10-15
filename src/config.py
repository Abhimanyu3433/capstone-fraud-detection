"""Loads the Colab hand-off (artifacts/) for the VS Code application.

Everything in Part 2 (training script, MLflow runs, FastAPI, tests) reads the
best model, its hyperparameters and the decision threshold from here instead
of hard-coding them.

Check the hand-off after copying it in:
    python -m src.config
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

REQUIRED_FILES = [
    "best_model.json",
    "feature_spec.json",
    "env_versions.json",
    "model_comparison.csv",
    "run_manifest.json",
    "models/fraud_pipeline_ci.pkl",
]
# Packages whose versions must match Colab for the pickle to load reliably.
PINNED_PACKAGES = ["scikit-learn", "imbalanced-learn", "xgboost", "numpy"]


@dataclass(frozen=True)
class BestModelConfig:
    model_key: str
    model_name: str
    estimator_class: str
    hyperparameters: dict
    imbalance: dict
    decision_threshold: float
    test_metrics: dict
    validation_metrics: dict
    pipeline_file: str
    ci_pipeline_file: str
    raw: dict

    @property
    def pipeline_path(self) -> Path:
        return ARTIFACTS_DIR / self.pipeline_file

    @property
    def ci_pipeline_path(self) -> Path:
        return ARTIFACTS_DIR / self.ci_pipeline_file


def _read_json(name: str) -> dict:
    path = ARTIFACTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run notebooks/01_colab_model_selection.ipynb on Colab and copy "
            "MyDrive/fraud-detection/artifacts/ into the project's artifacts/ folder."
        )
    return json.loads(path.read_text())


@lru_cache
def load_best_model_config() -> BestModelConfig:
    d = _read_json("best_model.json")
    return BestModelConfig(
        model_key=d["model_key"],
        model_name=d["model_name"],
        estimator_class=d["estimator_class"],
        hyperparameters=d["hyperparameters"],
        imbalance=d["imbalance"],
        decision_threshold=float(d["decision_threshold"]),
        test_metrics=d["test_metrics"],
        validation_metrics=d["validation_metrics"],
        pipeline_file=d["pipeline_file"],
        ci_pipeline_file=d["ci_pipeline_file"],
        raw=d,
    )


@lru_cache
def load_feature_spec() -> dict:
    return _read_json("feature_spec.json")


def check_environment() -> list[str]:
    """Return warnings for packages whose local version differs from Colab's."""
    colab = _read_json("env_versions.json")
    issues = []
    for pkg in PINNED_PACKAGES:
        want = colab.get(pkg)
        if not want:
            continue
        try:
            have = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            issues.append(f"{pkg}: not installed (Colab used {want})")
            continue
        if have != want:
            issues.append(f"{pkg}: local {have} != Colab {want}")
    return issues


def validate_handoff() -> None:
    missing = [f for f in REQUIRED_FILES if not (ARTIFACTS_DIR / f).exists()]
    if missing:
        print("Missing hand-off files:", *missing, sep="\n  - ")
        raise SystemExit(1)
    cfg = load_best_model_config()
    spec = load_feature_spec()
    print(f"Best model      : {cfg.model_name} ({cfg.estimator_class})")
    print(f"Hyperparameters : {cfg.hyperparameters}")
    print(f"Imbalance       : {cfg.imbalance}")
    print(f"Threshold       : {cfg.decision_threshold:.4f}")
    tm = cfg.test_metrics
    print(f"Test metrics    : P={tm['precision']:.4f} R={tm['recall']:.4f} F1={tm['f1']:.4f} "
          f"PR-AUC={tm['pr_auc']:.4f} ROC-AUC={tm['roc_auc']:.4f}")
    print(f"API inputs      : {spec['input_features']}")
    print(f"Full pipeline   : {'present' if cfg.pipeline_path.exists() else 'not copied (optional)'}")
    issues = check_environment()
    if issues:
        print("\nVersion mismatches:", *issues, sep="\n  - ")
        print("Fix: python -m src.config --pin  then  pip install -r requirements.txt")
    else:
        print("\nEnvironment matches Colab versions.")


def pin_requirements(req_file: Path = PROJECT_ROOT / "requirements-ci.txt") -> None:
    """Rewrite `pkg>=x` lines in requirements-ci.txt to `pkg==<Colab version>`."""
    colab = _read_json("env_versions.json")
    lines = req_file.read_text().splitlines()
    out = []
    for line in lines:
        name = line.split("[")[0].split(">")[0].split("=")[0].split("<")[0].strip()
        if name in PINNED_PACKAGES + ["pandas", "scipy", "joblib"] and colab.get(name):
            out.append(f"{name}=={colab[name]}")
        else:
            out.append(line)
    req_file.write_text("\n".join(out) + "\n")
    print(f"Pinned versions in {req_file.name} from artifacts/env_versions.json")


if __name__ == "__main__":
    import sys

    if "--pin" in sys.argv:
        pin_requirements()
    else:
        validate_handoff()
