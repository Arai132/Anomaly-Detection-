# Anomaly Detection Framework

A modular Python framework for detecting anomalies in tabular and time series data. Includes 12 detector implementations, a smart recommendation engine, and an interactive Streamlit UI.

Inspired by Chuying Ma's semi-supervised ensemble stacking approach (ODSC West 2023).

## Features

- **12 detectors** spanning statistical, density-based, deep learning, and ensemble methods
- **Smart recommender** — describe your data's nature and get ranked method suggestions
- **Streamlit UI** — upload data, get recommendations, tune parameters, and visualise results interactively
- **Semi-supervised mode** — works with as few as 10% labeled anomalies via iterative pseudo-label augmentation
- **Config-driven** — all hyperparameters in `config.yaml`, no magic numbers in source

## Detectors

| Detector | Best For |
|---|---|
| Z-Score | Univariate Gaussian, point anomalies |
| IQR | Skewed or unknown distributions |
| Isolation Forest | High-dimensional tabular data, mixed distributions |
| Local Outlier Factor (LOF) | Clustered / multimodal data |
| Autoencoder | High-dimensional correlated features |
| Gaussian Mixture Model | Probabilistic scoring, multimodal clusters |
| Elliptic Envelope | Gaussian multivariate with correlated features |
| One-Class SVM | Novelty detection on clean training data |
| PCA Reconstruction | Correlated high-dimensional data |
| Rolling Z-Score | Stationary or slowly drifting time series |
| STL Decomposition | Time series with trend and/or seasonality |
| Ensemble Stacking | Any data — combines all detectors for robustness |

## Quick Start

```bash
pip install -r requirements.txt
```

**Run the Streamlit UI:**
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

**Run the batch benchmark:**
```bash
python main.py
```
Trains all detectors on synthetic data and prints a comparison table. Plots saved to `reports/`.

**Run tests:**
```bash
pytest tests/test_detectors.py -v
```

## Project Structure

```
├── app.py                  # Streamlit interactive UI
├── main.py                 # Batch benchmark runner
├── config.yaml             # All hyperparameters
├── src/
│   ├── detectors/          # All detector implementations
│   │   ├── base.py         # BaseDetector ABC
│   │   ├── statistical.py  # ZScore, IQR
│   │   ├── isolation_forest.py
│   │   ├── lof.py
│   │   ├── autoencoder.py
│   │   ├── gaussian.py
│   │   ├── elliptic_envelope.py
│   │   ├── one_class_svm.py
│   │   ├── pca_detector.py
│   │   ├── timeseries.py   # RollingZScore, STL
│   │   └── ensemble_stacking.py
│   ├── recommender.py      # Nature taxonomy + scoring engine
│   └── utils/
│       ├── data_loader.py  # CSV/JSON loading, synthetic data generation
│       ├── evaluation.py   # Metrics (precision, recall, F1, ROC-AUC)
│       └── semi_supervised.py  # Pseudo-label augmentation loop
└── tests/
    └── test_detectors.py
```

## Streamlit UI Walkthrough

**Tab 1 — Upload Data**
Upload a CSV, JSON, TXT, or TSV file, or paste raw numbers directly. Optionally select a ground-truth label column (0 = normal, 1 = anomaly). Click **Auto-detect** to have the app heuristically infer your data's characteristics.

**Tab 2 — Describe & Recommend**
Select your data's nature across six categories — structure, distribution shape, temporal pattern, dimensionality, expected anomaly type, and label availability. The recommendation engine scores all 12 detectors against your selections and shows ranked suggestions with explanations.

**Tab 3 — Run & Results**
Choose which detectors to run, tune their parameters inline, and click **Run**. Results include anomaly score plots, a metrics comparison table (when labels are available), expandable anomaly row viewer, and a CSV download of all predictions.

## Detector Interface

All detectors share a common interface via `BaseDetector`:

```python
from src.detectors import IsolationForestDetector
from src.utils.data_loader import load_csv, preprocess

X, y = load_csv("data/my_data.csv", target_col="label")
X_train_s, X_test_s, scaler = preprocess(X_train, X_test)

det = IsolationForestDetector(n_estimators=200, contamination=0.03)
det.fit(X_train_s)
preds  = det.predict(X_test_s)       # 1 = anomaly, 0 = normal
scores = det.score_samples(X_test_s) # higher = more anomalous

det.save("models/iforest.pkl")
```

## Semi-Supervised Ensemble

For datasets with scarce labels, the ensemble stacking system iteratively expands the label set:

```python
from src.detectors import EnsembleStackingDetector, AutoencoderDetector, GaussianDetector, IsolationForestDetector
from src.utils.semi_supervised import make_sparse_labels, SemiSupervisedAugmenter

ensemble = EnsembleStackingDetector(detectors=[
    AutoencoderDetector(), GaussianDetector(), IsolationForestDetector()
])

# Simulate only 10% of anomalies being labeled
y_sparse = make_sparse_labels(y_train, label_fraction=0.1)

augmenter = SemiSupervisedAugmenter(detector=ensemble, max_iters=3)
y_augmented = augmenter.fit_augment(X_train, y_sparse)

ensemble.fit(X_train, y_augmented)
preds = ensemble.predict(X_test)
```

## Adding a New Detector

1. Subclass `BaseDetector` in `src/detectors/`
2. Implement `fit()`, `predict()`, and `score_samples()`
3. Export from `src/detectors/__init__.py`
4. Add hyperparameters under `detectors:` in `config.yaml`

## Requirements

- Python 3.12+
- scikit-learn, numpy, pandas, scipy, statsmodels
- streamlit, plotly
- See `requirements.txt` for full list
