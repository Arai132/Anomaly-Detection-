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

    def __init__(self, kernel: str = "rbf", nu: float = 0.05, gamma: str = "scale"):
        super().__init__(contamination=nu)
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma

    def fit(self, X: pd.DataFrame | np.ndarray, y=None) -> "OneClassSVMDetector":
        self._model = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma)
        self._model.fit(self._to_array(X))
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        raw = self._model.predict(self._to_array(X))
        return (raw == -1).astype(int)

    def score_samples(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return -self._model.decision_function(self._to_array(X))
