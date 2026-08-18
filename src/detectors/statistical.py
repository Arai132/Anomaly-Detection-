import numpy as np
import pandas as pd
from .base import BaseDetector


class ZScoreDetector(BaseDetector):
    """Flags points where any feature exceeds `threshold` standard deviations from the mean."""

    def __init__(self, threshold: float = 3.0):
        super().__init__()
        self.threshold = threshold
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y=None) -> "ZScoreDetector":
        arr = self._to_array(X)
        self._mean = arr.mean(axis=0)
        self._std = arr.std(axis=0)
        self._std[self._std == 0] = 1  # avoid division by zero
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        scores = self.score_samples(X)
        return (scores > self.threshold).astype(int)

    def score_samples(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        arr = self._to_array(X)
        z = np.abs((arr - self._mean) / self._std)
        return z.max(axis=1)


class IQRDetector(BaseDetector):
    """Flags points outside [Q1 - k*IQR, Q3 + k*IQR] on any feature."""

    def __init__(self, multiplier: float = 1.5):
        super().__init__()
        self.multiplier = multiplier
        self._q1: np.ndarray | None = None
        self._q3: np.ndarray | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y=None) -> "IQRDetector":
        arr = self._to_array(X)
        self._q1 = np.percentile(arr, 25, axis=0)
        self._q3 = np.percentile(arr, 75, axis=0)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        arr = self._to_array(X)
        iqr = self._q3 - self._q1
        lower = self._q1 - self.multiplier * iqr
        upper = self._q3 + self.multiplier * iqr
        outside = (arr < lower) | (arr > upper)
        return outside.any(axis=1).astype(int)

    def score_samples(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        arr = self._to_array(X)
        iqr = self._q3 - self._q1
        iqr[iqr == 0] = 1
        lower = self._q1 - self.multiplier * iqr
        upper = self._q3 + self.multiplier * iqr
        dist_lower = np.maximum(0, lower - arr) / iqr
        dist_upper = np.maximum(0, arr - upper) / iqr
        return np.maximum(dist_lower, dist_upper).max(axis=1)
