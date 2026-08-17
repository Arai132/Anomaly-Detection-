import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from .base import BaseDetector


class GaussianDetector(BaseDetector):
    """
    Gaussian Mixture Model detector.
    Anomaly score = negative log-likelihood (higher = more anomalous).
    Threshold set at `threshold_percentile` of training scores.
    """

    def __init__(
        self,
        n_components: int = 1,
        covariance_type: str = "full",
        threshold_percentile: float = 95,
        random_state: int = 42,
    ):
        super().__init__()
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.threshold_percentile = threshold_percentile
        self.random_state = random_state
        self._threshold: float | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y=None) -> "GaussianDetector":
        arr = self._to_array(X)
        self._model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
        )
        self._model.fit(arr)
        scores = self.score_samples(X)
        self._threshold = float(np.percentile(scores, self.threshold_percentile))
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return (self.score_samples(X) > self._threshold).astype(int)

    def score_samples(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        arr = self._to_array(X)
        return -self._model.score_samples(arr)  # negate: higher = more anomalous
