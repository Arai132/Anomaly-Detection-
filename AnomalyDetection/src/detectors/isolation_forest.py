import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from .base import BaseDetector


class IsolationForestDetector(BaseDetector):
    def __init__(self, n_estimators: int = 100, contamination: float = 0.05, random_state: int = 42):
        super().__init__(contamination)
        self._model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )

    def fit(self, X: pd.DataFrame | np.ndarray) -> "IsolationForestDetector":
        self._model.fit(self._to_array(X))
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        # sklearn uses -1/1; convert to 1=anomaly, 0=normal
        raw = self._model.predict(self._to_array(X))
        return (raw == -1).astype(int)

    def score_samples(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        # negate so higher = more anomalous
        return -self._model.score_samples(self._to_array(X))
