from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import joblib
from pathlib import Path


class BaseDetector(ABC):
    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self._model = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray | None = None) -> "BaseDetector":
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Returns 1 for anomaly, 0 for normal."""
        pass

    def fit_predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.fit(X).predict(X)

    def score_samples(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Returns anomaly scores (higher = more anomalous)."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support score_samples")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "BaseDetector":
        return joblib.load(path)

    def _to_array(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.values
        return np.asarray(X)
