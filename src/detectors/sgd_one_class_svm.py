import numpy as np
import pandas as pd
from sklearn.linear_model import SGDOneClassSVM
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from .base import BaseDetector


class SGDOneClassSVMDetector(BaseDetector):
    """
    Linear-time approximation of a kernel One-Class SVM: a Nystroem RBF
    feature map followed by SGDOneClassSVM. Scales to far larger datasets
    than OneClassSVMDetector's kernel SVM (O(n) vs O(n^2)-O(n^3) training
    cost), at some accuracy cost from the kernel approximation.
    """

    def __init__(
        self,
        nu: float = 0.05,
        gamma: str = "scale",
        n_components: int = 100,
        threshold_percentile: float | None = None,
        random_state: int = 42,
    ):
        super().__init__(contamination=nu)
        self.nu = nu
        self.gamma = gamma
        self.n_components = n_components
        self.threshold_percentile = threshold_percentile
        self.random_state = random_state
        self._threshold: float | None = None

    def _resolve_gamma(self, arr: np.ndarray) -> float:
        if self.gamma == "scale":
            return 1.0 / (arr.shape[1] * arr.var())
        return float(self.gamma)

    def fit(self, X: pd.DataFrame | np.ndarray, y=None) -> "SGDOneClassSVMDetector":
        arr = self._to_array(X)
        gamma = self._resolve_gamma(arr)
        self._model = make_pipeline(
            Nystroem(gamma=gamma, n_components=min(self.n_components, len(arr)), random_state=self.random_state),
            SGDOneClassSVM(nu=self.nu, random_state=self.random_state),
        )
        self._model.fit(arr)
        # SGD's approximate optimization doesn't reliably converge to a decision
        # boundary matching nu at extreme values (observed: 0 anomalies flagged
        # out of 85k rows at nu=0.0017 despite a real score ranking, 0.93 AUC).
        # Derive our own threshold from the score distribution instead of trusting
        # the model's own decision_function sign.
        pct = self.threshold_percentile if self.threshold_percentile is not None else (100 - self.contamination * 100)
        scores = self.score_samples(arr)
        self._threshold = float(np.percentile(scores, pct))
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return (self.score_samples(X) > self._threshold).astype(int)

    def score_samples(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return -self._model.decision_function(self._to_array(X))
