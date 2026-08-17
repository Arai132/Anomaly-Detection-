import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from .base import BaseDetector


class PCADetector(BaseDetector):
    """
    Projects data into a lower-dimensional PCA space and measures reconstruction error.
    Best for high-dimensional, correlated data. Anomalies = high reconstruction error.
    """

    def __init__(self, n_components: float = 0.95, threshold_percentile: float = 95):
        super().__init__()
        self.n_components = n_components  # float = variance explained; int = fixed components
        self.threshold_percentile = threshold_percentile
        self._threshold: float | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y=None) -> "PCADetector":
        arr = self._to_array(X)
        max_components = min(arr.shape)
        n = self.n_components
        if isinstance(n, float) and n < 1.0:
            n = min(max_components, max(1, int(n * arr.shape[1])))
        n = min(int(n), max_components)
        self._model = PCA(n_components=n)
        self._model.fit(arr)
        scores = self._reconstruction_error(arr)
        self._threshold = float(np.percentile(scores, self.threshold_percentile))
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return (self.score_samples(X) > self._threshold).astype(int)

    def score_samples(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self._reconstruction_error(self._to_array(X))

    def _reconstruction_error(self, arr: np.ndarray) -> np.ndarray:
        projected = self._model.transform(arr)
        reconstructed = self._model.inverse_transform(projected)
        return np.mean((arr - reconstructed) ** 2, axis=1)
