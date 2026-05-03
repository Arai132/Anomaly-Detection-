import numpy as np
import pytest
from src.utils.data_loader import generate_synthetic, preprocess
from src.detectors import IsolationForestDetector, LOFDetector, ZScoreDetector, IQRDetector
from src.detectors import AutoencoderDetector, GaussianDetector, EnsembleStackingDetector
from src.utils.semi_supervised import make_sparse_labels, SemiSupervisedAugmenter


@pytest.fixture
def data():
    X, y = generate_synthetic(n_samples=300, n_features=4, contamination=0.1)
    X_scaled, _, _ = preprocess(X)
    return X_scaled, y


def _check_detector(detector, X, y):
    detector.fit(X)
    preds = detector.predict(X)
    assert preds.shape == (len(X),)
    assert set(preds).issubset({0, 1})


def test_isolation_forest(data):
    X, y = data
    _check_detector(IsolationForestDetector(contamination=0.1), X, y)


def test_lof(data):
    X, y = data
    _check_detector(LOFDetector(contamination=0.1), X, y)


def test_zscore(data):
    X, y = data
    _check_detector(ZScoreDetector(threshold=2.5), X, y)


def test_iqr(data):
    X, y = data
    _check_detector(IQRDetector(multiplier=1.5), X, y)


def test_score_samples(data):
    X, y = data
    det = IsolationForestDetector()
    det.fit(X)
    scores = det.score_samples(X)
    assert scores.shape == (len(X),)
    assert scores.min() >= 0


def test_autoencoder(data):
    X, y = data
    _check_detector(AutoencoderDetector(encoding_dim=4, epochs=10), X, y)


def test_autoencoder_score_samples(data):
    X, y = data
    det = AutoencoderDetector(encoding_dim=4, epochs=10)
    det.fit(X)
    scores = det.score_samples(X)
    assert scores.shape == (len(X),)
    assert scores.min() >= 0


def test_gaussian(data):
    X, y = data
    _check_detector(GaussianDetector(n_components=1), X, y)


def test_gaussian_score_samples(data):
    X, y = data
    det = GaussianDetector()
    det.fit(X)
    scores = det.score_samples(X)
    assert scores.shape == (len(X),)


def test_ensemble_stacking_unsupervised(data):
    X, y = data
    detectors = [IsolationForestDetector(contamination=0.1), GaussianDetector()]
    ensemble = EnsembleStackingDetector(detectors=detectors)
    _check_detector(ensemble, X, y)


def test_ensemble_stacking_semi_supervised(data):
    X, y = data
    detectors = [IsolationForestDetector(contamination=0.1), GaussianDetector()]
    ensemble = EnsembleStackingDetector(detectors=detectors)
    y_sparse = make_sparse_labels(y, label_fraction=0.2)
    ensemble.fit(X, y_sparse)
    preds = ensemble.predict(X)
    assert preds.shape == (len(X),)
    assert set(preds).issubset({0, 1})


def test_semi_supervised_augmenter(data):
    X, y = data
    X_arr = X.values
    y_sparse = make_sparse_labels(y, label_fraction=0.1)
    assert (y_sparse == -1).sum() > 0, "should have unlabeled points"

    detectors = [IsolationForestDetector(contamination=0.1), GaussianDetector()]
    ensemble = EnsembleStackingDetector(detectors=detectors)
    augmenter = SemiSupervisedAugmenter(detector=ensemble, max_iters=2)
    y_aug = augmenter.fit_augment(X_arr, y_sparse)

    assert y_aug.shape == y.shape
    assert set(y_aug).issubset({-1, 0, 1})
    # augmentation should have labeled more points than the initial sparse set
    assert (y_aug != -1).sum() >= (y_sparse != -1).sum()
