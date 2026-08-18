"""
Real-world benchmark: fits all 10 tabular-compatible detectors plus the
EnsembleStackingDetector on the ULB Credit Card Fraud Detection dataset,
compares precision/recall/f1/ROC-AUC, and persists the ensemble as the
serving artifact used by api.py.

RollingZScoreDetector and STLDetector are excluded — they require a single
regularly-sampled time series, which this transaction data isn't.
"""
import warnings
import joblib
import yaml
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.utils.data_loader import load_fraud_dataset, preprocess
from src.utils.evaluation import evaluate, compare_detectors
from src.detectors import (
    IsolationForestDetector, LOFDetector, ZScoreDetector, IQRDetector,
    AutoencoderDetector, VAEDetector, GaussianDetector,
    EllipticEnvelopeDetector, SGDOneClassSVMDetector, PCADetector,
    EnsembleStackingDetector,
)
from src.visualization.plots import plot_confusion_matrix, plot_roc_curves

warnings.filterwarnings("ignore")

CONFIG_PATH = "config.yaml"
MODEL_PATH = "models/fraud_pipeline.joblib"


def build_detectors(cfg: dict, contamination: float, threshold_percentile: float) -> dict:
    d = cfg["detectors"]
    return {
        "IsolationForest": IsolationForestDetector(
            n_estimators=d["isolation_forest"]["n_estimators"],
            contamination=contamination,
            random_state=d["isolation_forest"]["random_state"],
            n_jobs=d["isolation_forest"]["n_jobs"],
        ),
        "LOF": LOFDetector(
            n_neighbors=d["lof"]["n_neighbors"],
            contamination=contamination,
            n_jobs=d["lof"]["n_jobs"],
        ),
        "ZScore": ZScoreDetector(threshold=d["zscore"]["threshold"]),
        "IQR": IQRDetector(multiplier=d["iqr"]["multiplier"]),
        "Autoencoder": AutoencoderDetector(
            encoding_dim=d["autoencoder"]["encoding_dim"],
            hidden_dim=d["autoencoder"]["hidden_dim"],
            epochs=d["autoencoder"]["epochs"],
            batch_size=d["autoencoder"]["batch_size"],
            lr=d["autoencoder"]["lr"],
            threshold_percentile=threshold_percentile,
            device=d["autoencoder"]["device"],
        ),
        "VAE": VAEDetector(
            latent_dim=d["vae"]["latent_dim"],
            hidden_dim=d["vae"]["hidden_dim"],
            epochs=d["vae"]["epochs"],
            batch_size=d["vae"]["batch_size"],
            lr=d["vae"]["lr"],
            beta=d["vae"]["beta"],
            threshold_percentile=threshold_percentile,
            device=d["vae"]["device"],
        ),
        "Gaussian": GaussianDetector(
            n_components=d["gaussian"]["n_components"],
            covariance_type=d["gaussian"]["covariance_type"],
            threshold_percentile=threshold_percentile,
        ),
        "EllipticEnvelope": EllipticEnvelopeDetector(
            contamination=contamination,
            support_fraction=d["elliptic_envelope"]["support_fraction"],
            random_state=d["elliptic_envelope"]["random_state"],
        ),
        "SGD-OneClassSVM": SGDOneClassSVMDetector(
            nu=contamination,
            gamma=d["sgd_one_class_svm"]["gamma"],
            n_components=d["sgd_one_class_svm"]["n_components"],
            random_state=d["sgd_one_class_svm"]["random_state"],
        ),
        "PCA": PCADetector(
            n_components=d["pca"]["n_components"],
            threshold_percentile=threshold_percentile,
        ),
    }


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    fb_cfg = cfg["fraud_benchmark"]

    print("Fetching real fraud-detection dataset (OpenML id=1597)...")
    X, y = load_fraud_dataset()
    print(f"Loaded {len(X)} transactions, {y.sum()} frauds ({y.mean():.4%} positive rate)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=fb_cfg["test_size"], random_state=fb_cfg["random_state"], stratify=y
    )

    max_n = fb_cfg["max_train_samples"]
    if len(X_train) > max_n:
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=max_n, random_state=fb_cfg["random_state"], stratify=y_train
        )
    print(f"Training on {len(X_train)} samples ({y_train.sum()} frauds), "
          f"testing on {len(X_test)} held-out samples ({y_test.sum()} frauds)")

    X_train_scaled, X_test_scaled, scaler = preprocess(X_train, X_test)

    contamination = float(y_train.mean())
    threshold_percentile = 100 - contamination * 100
    detectors = build_detectors(cfg, contamination, threshold_percentile)

    results = {}
    roc_data = {}
    for name, detector in detectors.items():
        print(f"Fitting {name}...")
        detector.fit(X_train_scaled)
        y_pred = detector.predict(X_test_scaled)
        try:
            scores = detector.score_samples(X_test_scaled)
        except NotImplementedError:
            scores = None

        metrics = evaluate(y_test, y_pred, scores)
        results[name] = metrics
        if scores is not None:
            roc_data[name] = {"y_true": y_test, "scores": scores, "roc_auc": metrics.get("roc_auc")}

    print("\n=== Base Detector Comparison (real fraud data) ===")
    print(compare_detectors(results).to_string())

    print("\nFitting EnsembleStacking (logistic-regression meta-learner over all 10 base detectors)...")
    ensemble = EnsembleStackingDetector(
        detectors=list(build_detectors(cfg, contamination, threshold_percentile).values()),
        threshold_percentile=threshold_percentile,
    )
    ensemble.fit(X_train_scaled, y_train)
    y_pred_ens = ensemble.predict(X_test_scaled)
    scores_ens = ensemble.score_samples(X_test_scaled)
    ensemble_metrics = evaluate(y_test, y_pred_ens, scores_ens)
    results["EnsembleStacking"] = ensemble_metrics
    roc_data["EnsembleStacking"] = {"y_true": y_test, "scores": scores_ens, "roc_auc": ensemble_metrics.get("roc_auc")}

    print("\n=== Final Comparison (incl. Ensemble) ===")
    comparison = compare_detectors(results)
    print(comparison.to_string())

    beats_all = all(
        ensemble_metrics["precision"] >= results[name]["precision"] and
        ensemble_metrics["recall"] >= results[name]["recall"]
        for name in detectors
    )
    print(f"\nEnsemble precision & recall >= every base detector: {beats_all}")

    Path("reports").mkdir(exist_ok=True)
    comparison.to_csv("reports/fraud_benchmark_results.csv")
    plot_confusion_matrix(y_test, y_pred_ens, title="EnsembleStacking — Fraud Benchmark", save_path="reports/fraud_EnsembleStacking_cm.png")
    plot_roc_curves(roc_data, save_path="reports/fraud_roc_curves.png")

    # Pin any PyTorch-backed base detectors (Autoencoder, VAE) to CPU before
    # pickling — a GPU-trained model saved with CUDA tensors can't be
    # deserialized on a GPU-less machine (e.g. the Docker containers this
    # artifact gets served from).
    for det in ensemble.detectors:
        if hasattr(det, "_model") and hasattr(det, "_device"):
            det._model.to("cpu")
            det._device = torch.device("cpu")

    Path("models").mkdir(exist_ok=True)
    joblib.dump(
        {"scaler": scaler, "model": ensemble, "feature_names": list(X_train.columns)},
        MODEL_PATH,
    )
    print(f"\nSaved serving artifact to {MODEL_PATH}")


if __name__ == "__main__":
    main()
