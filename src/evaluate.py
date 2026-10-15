"""Metrics, threshold selection and plots shared by training and MLflow tracking."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def f1_optimal_threshold(y_true, proba) -> float:
    """Threshold that maximises F1 on the given (validation) data."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    prec, rec = prec[:-1], rec[:-1]
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    return float(thr[int(np.argmax(f1))])


def metrics_at(y_true, proba, threshold: float) -> dict:
    """Classification metrics at a decision threshold (plus threshold-free AUCs)."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def plot_confusion_matrix(m: dict, title: str, path: str | Path) -> Path:
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["pred 0", "pred 1"])
    ax.set_yticks([0, 1], ["true 0", "true 1"])
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return Path(path)


def plot_roc_pr(fpr, tpr, precision, recall, title: str, path: str | Path, prevalence=None) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].plot(fpr, tpr)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC")
    axes[1].plot(recall, precision)
    if prevalence is not None:
        axes[1].axhline(prevalence, color="k", ls="--", lw=0.8)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return Path(path)


def curves_from_proba(y_true, proba) -> dict:
    fpr, tpr, _ = roc_curve(y_true, proba)
    prec, rec, _ = precision_recall_curve(y_true, proba)
    return {"roc": {"fpr": fpr, "tpr": tpr}, "pr": {"precision": prec, "recall": rec},
            "prevalence": float(np.mean(y_true))}


def plot_feature_importance(names, values, title: str, path: str | Path, top: int = 15) -> Path:
    order = np.argsort(values)[-top:]
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(order) + 1))
    ax.barh(np.asarray(names)[order], np.asarray(values)[order])
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return Path(path)


def model_feature_importance(pipeline):
    """(names, values) from a fitted pipeline's classifier, or None."""
    clf = pipeline.named_steps["clf"]
    names = list(pipeline.named_steps["prep"].get_feature_names_out())
    if hasattr(clf, "feature_importances_"):
        return names, np.asarray(clf.feature_importances_)
    if hasattr(clf, "coef_"):
        return names, np.abs(np.asarray(clf.coef_)[0])
    return None
