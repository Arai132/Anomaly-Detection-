import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from .base import BaseDetector


class IsolationForestDetector(BaseDetector):
    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.05,
        threshold_percentile: float | None = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        super().__init__(contamination)
        self.threshold_percentile = threshold_percentile
        self._model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self._threshold: float | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y=None) -> "IsolationForestDetector":
        arr = self._to_array(X)
        self._model.fit(arr)
        pct = self.threshold_percentile if self.threshold_percentile is not None else (100 - self.contamination * 100)
        scores = self.score_samples(arr)
        self._threshold = float(np.percentile(scores, pct))
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return (self.score_samples(X) > self._threshold).astype(int)

    def score_samples(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        # negate so higher = more anomalous
        return -self._model.score_samples(self._to_array(X))
