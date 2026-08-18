import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from .base import BaseDetector


class OneClassSVMDetector(BaseDetector):
    """
    Learns a decision boundary around normal data.
    Best for novelty detection on clean training data.
    Anomaly score = negative decision function (higher = more anomalous).
    """

    def __init__(
        self,
        kernel: str = "rbf",
        nu: float = 0.05,
        gamma: str = "scale",
        threshold_percentile: float | None = None,
    ):
        super().__init__(contamination=nu)
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.threshold_percentile = threshold_percentile
        self._threshold: float | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y=None) -> "OneClassSVMDetector":
        arr = self._to_array(X)
        self._model = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma)
        self._model.fit(arr)
        pct = self.threshold_percentile if self.threshold_percentile is not None else (100 - self.contamination * 100)
        scores = self.score_samples(arr)
        self._threshold = float(np.percentile(scores, pct))
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return (self.score_samples(X) > self._threshold).astype(int)

    def score_samples(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return -self._model.decision_function(self._to_array(X))
