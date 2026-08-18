import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from .base import BaseDetector


class EllipticEnvelopeDetector(BaseDetector):
    """
    Fits a robust covariance ellipse to the data.
    Best for Gaussian, unimodal, multivariate data.
    Anomaly score = negative Mahalanobis-like distance from the fitted ellipse.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        support_fraction: float | None = None,
        threshold_percentile: float | None = None,
        random_state: int = 42,
    ):
        super().__init__(contamination)
        self.support_fraction = support_fraction
        self.threshold_percentile = threshold_percentile
        self.random_state = random_state
        self._threshold: float | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y=None) -> "EllipticEnvelopeDetector":
        arr = self._to_array(X)
        self._model = EllipticEnvelope(
            contamination=self.contamination,
            support_fraction=self.support_fraction,
            random_state=self.random_state,
        )
        self._model.fit(arr)
        pct = self.threshold_percentile if self.threshold_percentile is not None else (100 - self.contamination * 100)
        scores = self.score_samples(arr)
        self._threshold = float(np.percentile(scores, pct))
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return (self.score_samples(X) > self._threshold).astype(int)

    def score_samples(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return -self._model.score_samples(self._to_array(X))
