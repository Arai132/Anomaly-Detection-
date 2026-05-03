"""Entry point: run all detectors on synthetic data and print a comparison table."""
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.data_loader import generate_synthetic, preprocess
from src.utils.evaluation import evaluate, compare_detectors
from src.utils.semi_supervised import make_sparse_labels, SemiSupervisedAugmenter
from src.detectors import (
    IsolationForestDetector, LOFDetector, ZScoreDetector, IQRDetector,
    AutoencoderDetector, GaussianDetector, EnsembleStackingDetector,
)
from src.visualization.plots import plot_anomaly_scores, plot_confusion_matrix, plot_roc_curves

CONFIG_PATH = "config.yaml"


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    print("Generating synthetic dataset...")
    X, y = generate_synthetic(n_samples=2000, n_features=5, contamination=0.05)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["data"]["test_size"], random_state=cfg["data"]["random_state"], stratify=y
    )
    X_train_scaled, X_test_scaled, _ = preprocess(X_train, X_test)

    detectors = {
        "IsolationForest": IsolationForestDetector(**cfg["detectors"]["isolation_forest"]),
        "LOF": LOFDetector(**cfg["detectors"]["lof"]),
        "ZScore": ZScoreDetector(threshold=cfg["detectors"]["zscore"]["threshold"]),
        "IQR": IQRDetector(multiplier=cfg["detectors"]["iqr"]["multiplier"]),
    }

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

        plot_confusion_matrix(y_test, y_pred, title=f"{name} — Confusion Matrix", save_path=f"reports/{name}_cm.png")
        if scores is not None:
            plot_anomaly_scores(scores, y_test, title=f"{name} — Anomaly Scores", save_path=f"reports/{name}_scores.png")

    if roc_data:
        plot_roc_curves(roc_data, save_path="reports/roc_curves.png")

    print("\n=== Detector Comparison ===")
    print(compare_detectors(results).to_string())

    # --- Semi-supervised ensemble stacking (Chuying Ma / ODSC West 2023) ---
    run_semi_supervised_ensemble(cfg, X_train_scaled, X_test_scaled, y_train, y_test)


def run_semi_supervised_ensemble(cfg, X_train, X_test, y_train, y_test):
    """
    Demonstrates the ensemble stacking approach:
    1. Build base detectors (deep learning, statistical, tree-based).
    2. Simulate scarce labels — only label_fraction of anomalies are known upfront.
    3. Run iterative pseudo-label augmentation to enrich labels.
    4. Fit ensemble with enriched labels; evaluate on held-out test set.
    """
    print("\n=== Semi-Supervised Ensemble Stacking ===")
    ss_cfg = cfg["semi_supervised"]
    es_cfg = cfg["detectors"]["ensemble_stacking"]
    ae_cfg = cfg["detectors"]["autoencoder"]
    gm_cfg = cfg["detectors"]["gaussian"]
    if_cfg = cfg["detectors"]["isolation_forest"]

    base_detectors = [
        AutoencoderDetector(**ae_cfg),
        GaussianDetector(**gm_cfg),
        IsolationForestDetector(**if_cfg),
    ]
    ensemble = EnsembleStackingDetector(
        detectors=base_detectors,
        threshold_percentile=es_cfg["threshold_percentile"],
    )

    # Simulate scarce labeling on training set
    y_sparse = make_sparse_labels(
        y_train,
        label_fraction=ss_cfg["label_fraction"],
        random_state=42,
    )
    n_labeled = int((y_sparse != -1).sum())
    n_anomalies_labeled = int((y_sparse == 1).sum())
    print(f"Initial labels: {n_labeled}/{len(y_train)} points "
          f"({n_anomalies_labeled} anomalies, {int((y_sparse == 0).sum())} normals)")

    # Iterative pseudo-label augmentation
    augmenter = SemiSupervisedAugmenter(
        detector=EnsembleStackingDetector(
            detectors=[
                AutoencoderDetector(**ae_cfg),
                GaussianDetector(**gm_cfg),
                IsolationForestDetector(**if_cfg),
            ],
            threshold_percentile=es_cfg["threshold_percentile"],
        ),
        anomaly_percentile=ss_cfg["anomaly_percentile"],
        normal_percentile=ss_cfg["normal_percentile"],
        max_iters=ss_cfg["max_iters"],
    )
    X_train_arr = X_train.values if hasattr(X_train, "values") else X_train
    X_test_arr = X_test.values if hasattr(X_test, "values") else X_test

    print("Running pseudo-label augmentation...")
    y_augmented = augmenter.fit_augment(X_train_arr, y_sparse)

    # Final ensemble fit with augmented labels, evaluate on test set
    ensemble.fit(X_train_arr, y_augmented)
    y_pred = ensemble.predict(X_test_arr)
    scores = ensemble.score_samples(X_test_arr)
    metrics = evaluate(y_test, y_pred, scores)

    print("\nEnsemble Stacking results (semi-supervised):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    plot_confusion_matrix(
        y_test, y_pred,
        title="EnsembleStacking (semi-supervised) — Confusion Matrix",
        save_path="reports/EnsembleStacking_cm.png",
    )
    plot_anomaly_scores(
        scores, y_test,
        title="EnsembleStacking — Unified Anomaly Scores",
        save_path="reports/EnsembleStacking_scores.png",
    )


if __name__ == "__main__":
    main()
