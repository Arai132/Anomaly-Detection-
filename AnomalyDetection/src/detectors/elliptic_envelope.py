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

    def __init__(self, contamination: float = 0.05, support_fraction: float | None = None, random_state: int = 42):
        super().__init__(contamination)
        self.support_fraction = support_fraction
        self.random_state = random_state

    def fit(self, X: pd.DataFrame | np.ndarray, y=None) -> "EllipticEnvelopeDetector":
        arr = self._to_array(X)
        self._model = EllipticEnvelope(
            contamination=self.contamination,
            support_fraction=self.support_fraction,
            random_state=self.random_state,
        )
        self._model.fit(arr)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        raw = self._model.predict(self._to_array(X))
        return (raw == -1).astype(int)

    def score_samples(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return -self._model.score_samples(self._to_array(X))
