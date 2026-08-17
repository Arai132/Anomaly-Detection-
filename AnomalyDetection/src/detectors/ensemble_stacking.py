import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from .base import BaseDetector


class EnsembleStackingDetector(BaseDetector):
    """
    Implements the ensemble stacking approach from Chuying Ma (ODSC West 2023).

    Each base detector produces an anomaly score. Scores are min-max normalized
    then stacked into a meta-feature matrix. Two modes:

    - Unsupervised: unified score = mean of normalized scores across detectors.
    - Semi-supervised: if y is provided to fit() (use -1 for unlabeled), a logistic
      regression meta-learner is trained on the labeled subset's stacked scores.

    All base detectors must implement score_samples().
    """

    def __init__(
        self,
        detectors: list[BaseDetector],
        threshold_percentile: float = 95,
    ):
        super().__init__()
        self.detectors = detectors
        self.threshold_percentile = threshold_percentile
        self._score_scaler = MinMaxScaler()
        self._meta_learner: LogisticRegression | None = None
        self._threshold: float | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray | None = None) -> "EnsembleStackingDetector":
        for det in self.detectors:
            det.fit(X)

        score_matrix = self._stack_scores(X)
        score_matrix_norm = self._score_scaler.fit_transform(score_matrix)

        if y is not None:
            labeled = y != -1
            y_labeled = y[labeled]
            # Only train meta-learner when both classes are present in labeled data
            if labeled.sum() >= 2 and len(np.unique(y_labeled)) > 1:
                self._meta_learner = LogisticRegression(
                    class_weight="balanced", random_state=42, max_iter=500
                )
                self._meta_learner.fit(score_matrix_norm[labeled], y_labeled)

        unified = self._unified_score(score_matrix_norm)
        self._threshold = float(np.percentile(unified, self.threshold_percentile))
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return (self.score_samples(X) > self._threshold).astype(int)

    def score_samples(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        score_matrix = self._stack_scores(X)
        score_matrix_norm = self._score_scaler.transform(score_matrix)
        return self._unified_score(score_matrix_norm)

    def _stack_scores(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return np.column_stack([det.score_samples(X) for det in self.detectors])

    def _unified_score(self, score_matrix_norm: np.ndarray) -> np.ndarray:
        if self._meta_learner is not None:
            return self._meta_learner.predict_proba(score_matrix_norm)[:, 1]
        return score_matrix_norm.mean(axis=1)
