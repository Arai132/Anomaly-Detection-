import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from .base import BaseDetector


class LOFDetector(BaseDetector):
    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.05,
        threshold_percentile: float | None = None,
        n_jobs: int = -1,
    ):
        super().__init__(contamination)
        self.n_neighbors = n_neighbors
        self.threshold_percentile = threshold_percentile
        self.n_jobs = n_jobs
        self._model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,
            n_jobs=n_jobs,
        )
        self._threshold: float | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y=None) -> "LOFDetector":
        arr = self._to_array(X)
        self._model.fit(arr)
        # sklearn's own contamination-derived offset_ can collapse to a
        # degenerate all-or-nothing cutoff at extreme contamination rates
        # (observed: 0 anomalies flagged out of 85k rows at 0.17% contamination).
        # Compute our own threshold from the score distribution instead, same
        # pattern as the other detectors that don't rely on sklearn's internal cutoff.
        pct = self.threshold_percentile if self.threshold_percentile is not None else (100 - self.contamination * 100)
        scores = self.score_samples(arr)
        self._threshold = float(np.percentile(scores, pct))
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return (self.score_samples(X) > self._threshold).astype(int)

    def score_samples(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return -self._model.score_samples(self._to_array(X))
