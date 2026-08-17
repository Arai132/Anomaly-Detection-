import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_anomaly_scores(scores: np.ndarray, labels: np.ndarray | None = None, title: str = "Anomaly Scores", save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(scores))
    ax.plot(x, scores, alpha=0.6, linewidth=0.8, label="score")
    if labels is not None:
        anomaly_idx = np.where(labels == 1)[0]
        ax.scatter(anomaly_idx, scores[anomaly_idx], color="red", s=20, zorder=5, label="anomaly")
    ax.set_title(title)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Anomaly score")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_feature_distributions(X: pd.DataFrame, labels: np.ndarray, max_features: int = 6, save_path: str | None = None):
    features = X.columns[:max_features]
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, features):
        normal = X.loc[labels == 0, col]
        anomalous = X.loc[labels == 1, col]
        ax.hist(normal, bins=40, alpha=0.6, label="normal", density=True)
        ax.hist(anomalous, bins=40, alpha=0.6, label="anomaly", density=True)
        ax.set_title(col)
        ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix", save_path: str | None = None):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_roc_curves(results: dict, save_path: str | None = None):
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, data in results.items():
        if "scores" in data and "y_true" in data:
            fpr, tpr, _ = roc_curve(data["y_true"], data["scores"])
            auc = data.get("roc_auc", "?")
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
