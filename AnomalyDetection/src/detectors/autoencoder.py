import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from .base import BaseDetector


class AutoencoderDetector(BaseDetector):
    """
    Reconstruction-error autoencoder using a single bottleneck hidden layer.
    Anomaly score = mean squared reconstruction error per sample.
    Threshold set at `threshold_percentile` of training scores.
    """

    def __init__(
        self,
        encoding_dim: int = 16,
        epochs: int = 50,
        batch_size: int = 32,
        threshold_percentile: float = 95,
        random_state: int = 42,
    ):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold_percentile = threshold_percentile
        self.random_state = random_state
        self._threshold: float | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y=None) -> "AutoencoderDetector":
        arr = self._to_array(X)
        self._model = MLPRegressor(
            hidden_layer_sizes=(self.encoding_dim,),
            max_iter=self.epochs,
            batch_size=min(self.batch_size, len(arr)),
            random_state=self.random_state,
            early_stopping=False,
            n_iter_no_change=self.epochs,  # disable early stopping heuristic
        )
        self._model.fit(arr, arr)
        scores = self._mse(arr)
        self._threshold = float(np.percentile(scores, self.threshold_percentile))
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return (self.score_samples(X) > self._threshold).astype(int)

    def score_samples(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        arr = self._to_array(X)
        return self._mse(arr)

    def _mse(self, arr: np.ndarray) -> np.ndarray:
        reconstructed = self._model.predict(arr)
        return np.mean((arr - reconstructed) ** 2, axis=1)
