import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from .base import BaseDetector


class LOFDetector(BaseDetector):
    def __init__(self, n_neighbors: int = 20, contamination: float = 0.05):
        super().__init__(contamination)
        self.n_neighbors = n_neighbors
        self._model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,
        )

    def fit(self, X: pd.DataFrame | np.ndarray) -> "LOFDetector":
        self._model.fit(self._to_array(X))
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        raw = self._model.predict(self._to_array(X))
        return (raw == -1).astype(int)

    def score_samples(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return -self._model.score_samples(self._to_array(X))
