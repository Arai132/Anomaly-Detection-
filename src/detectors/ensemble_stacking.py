import copy
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from .base import BaseDetector


class EnsembleStackingDetector(BaseDetector):
    """
    Implements the ensemble stacking approach from Chuying Ma (ODSC West 2023).

    Each base detector produces an anomaly score. Scores are min-max normalized
    then stacked into a meta-feature matrix. Two modes:

    - Unsupervised: unified score = mean of normalized scores across detectors.
    - Semi-supervised: if y is provided to fit() (use -1 for unlabeled), a logistic
      regression meta-learner is trained on the labeled subset's stacked scores.
      Those training scores are generated out-of-fold (5-fold CV over X) so the
      meta-learner never trains on a detector's in-sample score for a row it
      also fit on; the detectors actually used for inference are still fit on
      all of X.

    All base detectors must implement score_samples().
    """

    def __init__(
        self,
        detectors: list[BaseDetector],
        threshold_percentile: float = 95,
    ):
        super().__init__()
        self.detectors = detectors
        self.threshold_percentile = threshold_percentile
        self._score_scaler = MinMaxScaler()
        self._meta_learner: LogisticRegression | None = None
        self._threshold: float | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray | None = None) -> "EnsembleStackingDetector":
        meta_learner_trained = False

        if y is not None:
            labeled = y != -1
            y_labeled = y[labeled]
            # Only train meta-learner when both classes are present in labeled data
            if labeled.sum() >= 2 and len(np.unique(y_labeled)) > 1:
                # Fitting each detector on X and then scoring that same X would let
                # the meta-learner train on in-sample scores (each base detector
                # partly "remembers" the rows it was fit on) — that inflates
                # apparent separability during training and doesn't generalize.
                # Use out-of-fold scores instead, the standard fix for stacking.
                oof_matrix = self._out_of_fold_scores(X)
                oof_matrix_norm = self._score_scaler.fit_transform(oof_matrix)

                # L1 + moderate regularization so the meta-learner can zero out
                # detectors that carry no real signal on a given dataset, instead
                # of overfitting to their noise (visible with few positive labels).
                self._meta_learner = LogisticRegression(
                    class_weight="balanced", random_state=42, max_iter=2000,
                    l1_ratio=1.0, solver="liblinear", C=0.3,
                )
                self._meta_learner.fit(oof_matrix_norm[labeled], y_labeled)

                # class_weight="balanced" recalibrates predict_proba against an
                # assumed 50/50 split, so a raw percentile-of-score cutoff (correct
                # for the unsupervised path) badly overestimates flags under real,
                # heavily-imbalanced label rates. With labels available, pick the
                # threshold that maximizes F1 on the labeled subset instead.
                labeled_scores = self._meta_learner.predict_proba(oof_matrix_norm[labeled])[:, 1]
                precision, recall, thresholds = precision_recall_curve(y_labeled, labeled_scores)
                f1 = np.divide(
                    2 * precision * recall, precision + recall,
                    out=np.zeros_like(precision), where=(precision + recall) > 0,
                )
                self._threshold = float(thresholds[np.argmax(f1[:-1])])
                meta_learner_trained = True

        # Detectors used at inference time are always the ones fit on all of X,
        # regardless of whether the meta-learner used out-of-fold scores above.
        for det in self.detectors:
            det.fit(X)

        if not meta_learner_trained:
            score_matrix = self._stack_scores(X)
            score_matrix_norm = self._score_scaler.fit_transform(score_matrix)
            unified = self._unified_score(score_matrix_norm)
            self._threshold = float(np.percentile(unified, self.threshold_percentile))

        self.is_fitted = True
        return self

    def _out_of_fold_scores(self, X: pd.DataFrame | np.ndarray, n_splits: int = 5, random_state: int = 42) -> np.ndarray:
        arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        n_splits = min(n_splits, len(arr))
        oof = np.zeros((len(arr), len(self.detectors)))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for train_idx, holdout_idx in kf.split(arr):
            for j, det in enumerate(self.detectors):
                det_fold = copy.deepcopy(det)
                det_fold.fit(arr[train_idx])
                oof[holdout_idx, j] = det_fold.score_samples(arr[holdout_idx])
        return oof

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return (self.score_samples(X) > self._threshold).astype(int)

    def score_samples(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        score_matrix = self._stack_scores(X)
        score_matrix_norm = self._score_scaler.transform(score_matrix)
        return self._unified_score(score_matrix_norm)

    def _stack_scores(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return np.column_stack([det.score_samples(X) for det in self.detectors])

    def _unified_score(self, score_matrix_norm: np.ndarray) -> np.ndarray:
        if self._meta_learner is not None:
            return self._meta_learner.predict_proba(score_matrix_norm)[:, 1]
        return score_matrix_norm.mean(axis=1)
