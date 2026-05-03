# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run full pipeline (generates data, trains all detectors, saves plots to reports/)
python main.py

# Run all tests
pytest tests/test_detectors.py -v

# Run a single test
pytest tests/test_detectors.py::test_ensemble_stacking_semi_supervised -v
```

Python is invoked as `python` on Windows and `python3` on Linux. The project has no build or lint step configured.

## Architecture

All detectors extend `BaseDetector` (ABC in `src/detectors/base.py`). The interface is:
- `fit(X, y=None)` — trains and returns `self`
- `predict(X)` → `np.ndarray` of `0`/`1` where **1 = anomaly**
- `score_samples(X)` → float scores where **higher = more anomalous**

**Critical convention:** sklearn uses `−1` for anomaly; all detectors invert this to `1`. `score_samples()` is always negated relative to the underlying library so the direction is consistent across every detector.

## Config-driven design

All hyperparameters live in `config.yaml` — no magic numbers in source files. `main.py` reads the config and passes kwargs directly to detector constructors, so adding a new config key is enough to expose a new hyperparameter.

## Ensemble stacking system

`EnsembleStackingDetector` (`src/detectors/ensemble_stacking.py`) is the centrepiece added from the ODSC paper. It:

1. Fits N base detectors (one per detector family: deep learning, statistical, tree-based)
2. Collects each detector's `score_samples()` output into an `[N_samples × N_detectors]` matrix
3. Min-max normalises each column independently (fitted `MinMaxScaler` stored on `self`)
4. **Unsupervised:** unified score = row mean of normalised scores
5. **Semi-supervised:** if `y` is passed to `fit()` with `-1` for unlabeled rows, trains a `LogisticRegression` meta-learner on the labeled subset; its `predict_proba()` becomes the unified score

The threshold separating 0/1 predictions is always the `threshold_percentile` of training unified scores.

## Semi-supervised loop

`SemiSupervisedAugmenter` (`src/utils/semi_supervised.py`) iteratively expands a sparse label set:
- Label convention: `-1` = unlabeled, `0` = normal, `1` = anomaly
- Each iteration fits the ensemble on labeled rows, scores unlabeled rows, promotes high/low-scoring points to pseudo-labels
- `make_sparse_labels(y_true, label_fraction)` simulates a realistic scarce-label scenario for experiments

## Adding a new detector

1. Subclass `BaseDetector` in `src/detectors/`
2. Implement `fit()` and `predict()`; implement `score_samples()` if the detector produces continuous scores (required to participate in `EnsembleStackingDetector`)
3. Export from `src/detectors/__init__.py`
4. Add hyperparameters under `detectors:` in `config.yaml`

To include the new detector in the ensemble, pass it in the `detectors=[...]` list to `EnsembleStackingDetector`.

## Data utilities

- `generate_synthetic(n_samples, n_features, contamination)` — synthetic benchmark data; anomalies drawn from `Uniform(−6, 6)`, normals from `N(0,1)`
- `load_csv(path, target_col)` — loads a real CSV; returns `(X, y)` where `y` is `None` if the column is absent
- `preprocess(X_train, X_test)` — fits `StandardScaler` on train, transforms both; returns `(X_train_scaled, X_test_scaled, scaler)`
- Plots are saved to `reports/` (created automatically); `matplotlib` backend is non-interactive
