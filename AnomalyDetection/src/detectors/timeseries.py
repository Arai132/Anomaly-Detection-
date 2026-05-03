import numpy as np
import pandas as pd
from .base import BaseDetector


class RollingZScoreDetector(BaseDetector):
    """
    Rolling-window z-score for univariate time series.
    Flags points where |(x - rolling_mean) / rolling_std| > threshold.
    """

    def __init__(self, window: int = 20, threshold: float = 3.0, min_periods: int = 5):
        super().__init__()
        self.window = window
        self.threshold = threshold
        self.min_periods = min_periods

    def fit(self, X: pd.DataFrame | np.ndarray, y=None) -> "RollingZScoreDetector":
        # stateless: no training needed beyond storing params
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return (self.score_samples(X) > self.threshold).astype(int)

    def score_samples(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        arr = self._to_array(X)
        if arr.ndim > 1:
            arr = arr[:, 0]
        series = pd.Series(arr)
        roll_mean = series.rolling(self.window, min_periods=self.min_periods, center=True).mean()
        roll_std = series.rolling(self.window, min_periods=self.min_periods, center=True).std()
        roll_std = roll_std.replace(0, np.nan).fillna(1)
        z = np.abs((series - roll_mean) / roll_std)
        return z.fillna(0).values


class STLDetector(BaseDetector):
    """
    STL decomposition: trend + seasonal + residual.
    Anomaly score = |residual| normalised by its IQR.
    Best for time series with clear trend and/or seasonality.
    Requires statsmodels.
    """

    def __init__(self, period: int = 12, threshold_percentile: float = 95, robust: bool = True):
        super().__init__()
        self.period = period
        self.threshold_percentile = threshold_percentile
        self.robust = robust
        self._threshold: float | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y=None) -> "STLDetector":
        arr = self._to_array(X)
        if arr.ndim > 1:
            arr = arr[:, 0]
        self._train_arr = arr
        scores = self._residual_scores(arr)
        self._threshold = float(np.percentile(scores, self.threshold_percentile))
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return (self.score_samples(X) > self._threshold).astype(int)

    def score_samples(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        arr = self._to_array(X)
        if arr.ndim > 1:
            arr = arr[:, 0]
        return self._residual_scores(arr)

    def _residual_scores(self, arr: np.ndarray) -> np.ndarray:
        from statsmodels.tsa.seasonal import STL
        stl = STL(arr, period=self.period, robust=self.robust)
        result = stl.fit()
        residuals = result.resid
        q1, q3 = np.percentile(residuals, [25, 75])
        iqr = max(q3 - q1, 1e-10)
        return np.abs(residuals) / iqr
