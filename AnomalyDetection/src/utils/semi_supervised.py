import numpy as np


def make_sparse_labels(
    y_true: np.ndarray,
    label_fraction: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """
    Simulates scarce labeling: keeps `label_fraction` of each class labeled,
    marks the rest as -1 (unlabeled).
    """
    rng = np.random.default_rng(random_state)
    y_sparse = np.full(len(y_true), -1, dtype=float)

    for cls in np.unique(y_true):
        idx = np.where(y_true == cls)[0]
        n_keep = max(1, int(len(idx) * label_fraction))
        chosen = rng.choice(idx, n_keep, replace=False)
        y_sparse[chosen] = cls

    return y_sparse


class SemiSupervisedAugmenter:
    """
    Iterative pseudo-label augmentation loop.

    Each iteration:
      1. Fit the detector on the currently labeled subset.
      2. Score all unlabeled points.
      3. Assign label=1 to points above `anomaly_percentile` of unlabeled scores.
      4. Assign label=0 to points below `normal_percentile` of unlabeled scores.
      5. Repeat until no new labels are added or max_iters reached.

    y_sparse convention: -1 = unlabeled, 0 = normal, 1 = anomaly.
    """

    def __init__(
        self,
        detector,
        anomaly_percentile: float = 90,
        normal_percentile: float = 20,
        max_iters: int = 3,
    ):
        self.detector = detector
        self.anomaly_percentile = anomaly_percentile
        self.normal_percentile = normal_percentile
        self.max_iters = max_iters

    def fit_augment(self, X: np.ndarray, y_sparse: np.ndarray) -> np.ndarray:
        """Returns augmented labels; unlabeled points that remain ambiguous stay -1."""
        y = y_sparse.copy().astype(float)

        for i in range(self.max_iters):
            labeled_mask = y != -1
            unlabeled_mask = y == -1

            if labeled_mask.sum() == 0 or unlabeled_mask.sum() == 0:
                break

            self.detector.fit(X[labeled_mask], y[labeled_mask].astype(int))

            unlabeled_idx = np.where(unlabeled_mask)[0]
            scores = self.detector.score_samples(X[unlabeled_idx])

            hi = np.percentile(scores, self.anomaly_percentile)
            lo = np.percentile(scores, self.normal_percentile)

            new_anom = unlabeled_idx[scores >= hi]
            new_norm = unlabeled_idx[scores <= lo]

            y[new_anom] = 1
            y[new_norm] = 0

            print(f"  Iter {i + 1}: +{len(new_anom)} anomalies, +{len(new_norm)} normals "
                  f"({int((y != -1).sum())} total labeled)")

            if len(new_anom) + len(new_norm) == 0:
                break

        return y.astype(int)
