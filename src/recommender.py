"""
Recommendation engine: maps user-described data natures to ranked detector suggestions.

Scoring system per (detector, nature_value): -2 = poor fit, 0 = neutral, +3 = ideal.
"""
from __future__ import annotations
from dataclasses import dataclass, field

# ── Nature taxonomy ────────────────────────────────────────────────────────────

NATURE_TAXONOMY: dict[str, dict] = {
    "structure": {
        "label": "Data Structure",
        "help": "How is your data organised?",
        "single": True,
        "options": {
            "univariate":   "Univariate — single column / variable",
            "multivariate": "Multivariate — multiple independent features",
            "time_series":  "Time Series — ordered observations over time",
        },
    },
    "distribution": {
        "label": "Distribution Shape",
        "help": "What does a histogram of your data look like?",
        "single": False,
        "options": {
            "gaussian":     "Gaussian / Normal — symmetric bell curve",
            "skewed":       "Skewed — long right or left tail (e.g. counts, durations)",
            "heavy_tailed": "Heavy-tailed — extreme values occur more often than Gaussian",
            "multimodal":   "Multimodal — two or more distinct peaks/clusters",
            "uniform":      "Uniform — values spread roughly evenly across a range",
            "categorical":  "Categorical / Enumerated — finite discrete values",
            "sparse":       "Sparse — most values are zero or a single default",
            "unknown":      "Unknown / Unsure",
        },
    },
    "pattern": {
        "label": "Temporal / Structural Pattern",
        "help": "Does the data follow a recognisable pattern over time or space?",
        "single": False,
        "options": {
            "stationary":   "Stationary — stable mean and variance",
            "trending":     "Trending — steadily drifting up or down",
            "seasonal":     "Seasonal / Periodic — repeating cycles",
            "random_walk":  "Random Walk — each value depends on the previous",
            "bursty":       "Bursty — long quiet periods with sudden spikes",
            "correlated":   "Cross-correlated — multiple variables move together",
            "none":         "No specific pattern / Not applicable",
        },
    },
    "dimensionality": {
        "label": "Dimensionality",
        "help": "How many features/columns does the data have?",
        "single": True,
        "options": {
            "low":    "Low (1–5 features)",
            "medium": "Medium (6–20 features)",
            "high":   "High (21+ features)",
        },
    },
    "anomaly_type": {
        "label": "Expected Anomaly Type",
        "help": "What kind of anomaly are you looking for?",
        "single": False,
        "options": {
            "point":       "Point — single abnormal observation",
            "contextual":  "Contextual — normal in isolation, anomalous in context",
            "collective":  "Collective — a run or group of points anomalous together",
            "novelty":     "Novelty — a new pattern not seen during training",
        },
    },
    "labels": {
        "label": "Label Availability",
        "help": "Do you have any confirmed anomaly labels?",
        "single": True,
        "options": {
            "none": "No labels (fully unsupervised)",
            "few":  "A few confirmed labels (semi-supervised)",
        },
    },
}


# ── Detector registry ──────────────────────────────────────────────────────────

@dataclass
class DetectorSpec:
    id: str
    name: str
    short: str          # one-line tagline
    description: str    # longer explanation for the UI
    strengths: list[str]
    weaknesses: list[str]
    time_series_only: bool = False
    scores: dict[str, dict[str, int]] = field(default_factory=dict)
    params: dict = field(default_factory=dict)   # default init kwargs


DETECTORS: list[DetectorSpec] = [
    DetectorSpec(
        id="zscore",
        name="Z-Score",
        short="Classic 3-sigma rule for Gaussian data",
        description=(
            "Flags samples whose feature values deviate more than N standard deviations "
            "from the training mean. Simple, fast, and highly interpretable. Works best "
            "when each feature is independently Gaussian."
        ),
        strengths=["Extremely interpretable", "Fast", "Works well for univariate Gaussian data"],
        weaknesses=["Fails on skewed or heavy-tailed distributions", "Ignores correlations between features"],
        scores={
            "structure":      {"univariate": 3, "multivariate": 1, "time_series": 1},
            "distribution":   {"gaussian": 3, "skewed": -1, "heavy_tailed": -2, "multimodal": -1, "uniform": 0, "categorical": -2, "sparse": -1, "unknown": 0},
            "pattern":        {"stationary": 3, "trending": -1, "seasonal": -1, "random_walk": 0, "bursty": 1, "correlated": -1, "none": 1},
            "dimensionality": {"low": 3, "medium": 1, "high": -1},
            "anomaly_type":   {"point": 3, "contextual": -1, "collective": -2, "novelty": 0},
            "labels":         {"none": 2, "few": 0},
        },
        params={"threshold": 3.0},
    ),
    DetectorSpec(
        id="iqr",
        name="IQR (Interquartile Range)",
        short="Robust outlier detection using quartiles",
        description=(
            "Flags samples outside [Q1 − k·IQR, Q3 + k·IQR] on any feature. "
            "Robust to skewed distributions since it doesn't assume Gaussian data. "
            "A more resistant alternative to Z-Score."
        ),
        strengths=["Robust to skewed distributions", "No Gaussian assumption", "Simple to interpret"],
        weaknesses=["Ignores feature correlations", "Per-feature only, misses multivariate patterns"],
        scores={
            "structure":      {"univariate": 3, "multivariate": 1, "time_series": 1},
            "distribution":   {"gaussian": 2, "skewed": 3, "heavy_tailed": 2, "multimodal": 0, "uniform": 1, "categorical": -1, "sparse": 0, "unknown": 2},
            "pattern":        {"stationary": 2, "trending": 0, "seasonal": 0, "random_walk": 0, "bursty": 2, "correlated": -1, "none": 1},
            "dimensionality": {"low": 3, "medium": 1, "high": -1},
            "anomaly_type":   {"point": 3, "contextual": -1, "collective": -2, "novelty": 0},
            "labels":         {"none": 2, "few": 0},
        },
        params={"multiplier": 1.5},
    ),
    DetectorSpec(
        id="isolation_forest",
        name="Isolation Forest",
        short="Fast tree-based method — works on almost any tabular data",
        description=(
            "Randomly partitions features and scores anomalies by how few splits "
            "are needed to isolate them. Anomalies are 'few and different' — they "
            "are isolated quickly. Scales well to high dimensions and large datasets."
        ),
        strengths=["Handles high dimensions well", "Robust to irrelevant features", "Fast, scalable", "No distribution assumption"],
        weaknesses=["Misses contextual/sequential anomalies", "Less effective on very correlated data"],
        scores={
            "structure":      {"univariate": 2, "multivariate": 3, "time_series": 1},
            "distribution":   {"gaussian": 2, "skewed": 2, "heavy_tailed": 2, "multimodal": 2, "uniform": 1, "categorical": -1, "sparse": 2, "unknown": 3},
            "pattern":        {"stationary": 2, "trending": 1, "seasonal": 0, "random_walk": 0, "bursty": 2, "correlated": 1, "none": 2},
            "dimensionality": {"low": 2, "medium": 3, "high": 3},
            "anomaly_type":   {"point": 3, "contextual": 0, "collective": -1, "novelty": 1},
            "labels":         {"none": 3, "few": 1},
        },
        params={"n_estimators": 100, "contamination": 0.05},
    ),
    DetectorSpec(
        id="lof",
        name="Local Outlier Factor (LOF)",
        short="Density-based — detects outliers in clusters of varying density",
        description=(
            "Compares each point's local density to its neighbours. Points in sparse "
            "neighbourhoods compared to their neighbours are flagged. Excellent for "
            "datasets with multiple clusters of different densities."
        ),
        strengths=["Handles multimodal / clustered data", "Local perspective catches subtle outliers", "No global distribution assumption"],
        weaknesses=["Slow on large datasets", "Struggles with high dimensions (curse of dimensionality)", "Sensitive to n_neighbors"],
        scores={
            "structure":      {"univariate": 1, "multivariate": 3, "time_series": 0},
            "distribution":   {"gaussian": 2, "skewed": 2, "heavy_tailed": 1, "multimodal": 3, "uniform": 1, "categorical": -1, "sparse": 0, "unknown": 2},
            "pattern":        {"stationary": 2, "trending": 0, "seasonal": 0, "random_walk": 0, "bursty": 1, "correlated": 2, "none": 2},
            "dimensionality": {"low": 3, "medium": 2, "high": -1},
            "anomaly_type":   {"point": 3, "contextual": 1, "collective": 0, "novelty": -1},
            "labels":         {"none": 2, "few": 0},
        },
        params={"n_neighbors": 20, "contamination": 0.05},
    ),
    DetectorSpec(
        id="autoencoder",
        name="Autoencoder",
        short="Neural network reconstruction error — great for complex patterns",
        description=(
            "An MLP is trained to compress then reconstruct inputs through a bottleneck. "
            "Points the model cannot reconstruct well (high MSE) are anomalies. "
            "Learns non-linear feature interactions automatically."
        ),
        strengths=["Captures non-linear correlations", "Handles high-dimensional data", "Learns complex normal patterns"],
        weaknesses=["Needs sufficient training data", "Slower to train", "Less interpretable"],
        scores={
            "structure":      {"univariate": 0, "multivariate": 3, "time_series": 1},
            "distribution":   {"gaussian": 2, "skewed": 2, "heavy_tailed": 2, "multimodal": 2, "uniform": 1, "categorical": -1, "sparse": 1, "unknown": 2},
            "pattern":        {"stationary": 2, "trending": 1, "seasonal": 1, "random_walk": 0, "bursty": 1, "correlated": 3, "none": 1},
            "dimensionality": {"low": 0, "medium": 2, "high": 3},
            "anomaly_type":   {"point": 2, "contextual": 2, "collective": 2, "novelty": 2},
            "labels":         {"none": 2, "few": 1},
        },
        params={"encoding_dim": 16, "epochs": 50, "threshold_percentile": 95},
    ),
    DetectorSpec(
        id="vae",
        name="Variational Autoencoder (VAE)",
        short="Probabilistic reconstruction error — regularized latent space",
        description=(
            "Like an autoencoder, but the bottleneck is a learned distribution "
            "(mean + variance) instead of a fixed vector, regularized toward a "
            "standard normal prior via KL divergence. The smoother latent space "
            "often generalizes better than a plain autoencoder on complex data."
        ),
        strengths=["Regularized latent space generalizes well", "Captures non-linear correlations", "Handles high-dimensional data"],
        weaknesses=["More hyperparameters to tune (beta, latent_dim)", "Needs sufficient training data", "Less interpretable"],
        scores={
            "structure":      {"univariate": 0, "multivariate": 3, "time_series": 1},
            "distribution":   {"gaussian": 3, "skewed": 2, "heavy_tailed": 2, "multimodal": 2, "uniform": 1, "categorical": -1, "sparse": 1, "unknown": 2},
            "pattern":        {"stationary": 2, "trending": 1, "seasonal": 1, "random_walk": 0, "bursty": 1, "correlated": 3, "none": 1},
            "dimensionality": {"low": 0, "medium": 2, "high": 3},
            "anomaly_type":   {"point": 2, "contextual": 2, "collective": 2, "novelty": 2},
            "labels":         {"none": 2, "few": 1},
        },
        params={"latent_dim": 16, "epochs": 50, "beta": 1.0, "threshold_percentile": 95},
    ),
    DetectorSpec(
        id="gaussian",
        name="Gaussian Mixture Model (GMM)",
        short="Probabilistic model — anomalies have low likelihood",
        description=(
            "Fits a mixture of Gaussians and flags points with low log-likelihood. "
            "With n_components=1 it is equivalent to a full Gaussian; increasing components "
            "handles multimodal data. Provides interpretable probability scores."
        ),
        strengths=["Soft probabilistic scores", "Handles multimodal data (multiple components)", "Interpretable"],
        weaknesses=["Assumes Gaussian-like clusters", "Degrades in high dimensions", "Sensitive to n_components choice"],
        scores={
            "structure":      {"univariate": 2, "multivariate": 3, "time_series": 0},
            "distribution":   {"gaussian": 3, "skewed": -1, "heavy_tailed": -1, "multimodal": 3, "uniform": -1, "categorical": -2, "sparse": -1, "unknown": 1},
            "pattern":        {"stationary": 3, "trending": 0, "seasonal": 0, "random_walk": 0, "bursty": 1, "correlated": 2, "none": 2},
            "dimensionality": {"low": 3, "medium": 2, "high": -2},
            "anomaly_type":   {"point": 2, "contextual": 1, "collective": 0, "novelty": 1},
            "labels":         {"none": 2, "few": 0},
        },
        params={"n_components": 1, "threshold_percentile": 95},
    ),
    DetectorSpec(
        id="elliptic_envelope",
        name="Elliptic Envelope",
        short="Robust covariance ellipse for Gaussian multivariate data",
        description=(
            "Fits a robust covariance estimate (Minimum Covariance Determinant) and flags "
            "points far from the centre of the fitted ellipsoid. Directly models "
            "correlations between features under a Gaussian assumption."
        ),
        strengths=["Explicitly models feature correlations", "Robust to moderate contamination", "Well-calibrated for Gaussian data"],
        weaknesses=["Assumes single Gaussian cluster", "Fails on non-Gaussian distributions", "Breaks down in high dimensions"],
        scores={
            "structure":      {"univariate": 1, "multivariate": 3, "time_series": 0},
            "distribution":   {"gaussian": 3, "skewed": -2, "heavy_tailed": -2, "multimodal": -2, "uniform": -1, "categorical": -3, "sparse": -1, "unknown": 0},
            "pattern":        {"stationary": 3, "trending": -1, "seasonal": -1, "random_walk": -1, "bursty": 0, "correlated": 3, "none": 1},
            "dimensionality": {"low": 3, "medium": 2, "high": -2},
            "anomaly_type":   {"point": 3, "contextual": 0, "collective": -1, "novelty": 1},
            "labels":         {"none": 2, "few": 0},
        },
        params={"contamination": 0.05},
    ),
    DetectorSpec(
        id="one_class_svm",
        name="One-Class SVM",
        short="Boundary-learning method — best for novelty detection",
        description=(
            "Learns a tight boundary around normal training data using an RBF kernel. "
            "Anything outside that boundary at test time is flagged. Ideal when "
            "training data is clean and you want to detect truly new patterns."
        ),
        strengths=["Excellent novelty detection on clean data", "Flexible non-linear boundary", "Works for complex distributions"],
        weaknesses=["Sensitive to contamination in training data", "Slow on large datasets", "Requires careful nu/gamma tuning"],
        scores={
            "structure":      {"univariate": 1, "multivariate": 2, "time_series": 0},
            "distribution":   {"gaussian": 2, "skewed": 2, "heavy_tailed": 1, "multimodal": 1, "uniform": 1, "categorical": -1, "sparse": 0, "unknown": 2},
            "pattern":        {"stationary": 2, "trending": 0, "seasonal": 0, "random_walk": 0, "bursty": 1, "correlated": 2, "none": 1},
            "dimensionality": {"low": 2, "medium": 2, "high": 0},
            "anomaly_type":   {"point": 2, "contextual": 1, "collective": 0, "novelty": 3},
            "labels":         {"none": 2, "few": 0},
        },
        params={"nu": 0.05, "kernel": "rbf"},
    ),
    DetectorSpec(
        id="pca",
        name="PCA Reconstruction Error",
        short="Linear dimensionality reduction — best for correlated high-dim data",
        description=(
            "Projects data into a lower-dimensional PCA space and measures how well "
            "each point can be reconstructed. Points that don't fit the main linear "
            "structure have high reconstruction error and are flagged as anomalies."
        ),
        strengths=["Excellent for correlated high-dimensional data", "Fast and scalable", "Interpretable via explained variance"],
        weaknesses=["Only captures linear relationships", "Misses non-linear patterns", "Needs many correlated features to be useful"],
        scores={
            "structure":      {"univariate": -1, "multivariate": 3, "time_series": 1},
            "distribution":   {"gaussian": 2, "skewed": 1, "heavy_tailed": 0, "multimodal": -1, "uniform": 0, "categorical": -2, "sparse": 1, "unknown": 1},
            "pattern":        {"stationary": 2, "trending": 1, "seasonal": 0, "random_walk": 0, "bursty": 0, "correlated": 3, "none": 1},
            "dimensionality": {"low": -1, "medium": 2, "high": 3},
            "anomaly_type":   {"point": 2, "contextual": 1, "collective": 2, "novelty": 2},
            "labels":         {"none": 2, "few": 0},
        },
        params={"n_components": 0.95, "threshold_percentile": 95},
    ),
    DetectorSpec(
        id="ensemble_stacking",
        name="Ensemble Stacking",
        short="Combines all base detectors — most robust, semi-supervised capable",
        description=(
            "Runs Autoencoder + GMM + Isolation Forest, normalises their scores, "
            "then fuses them via a mean (unsupervised) or logistic regression "
            "meta-learner (when a few labels are available). Reduces false positives "
            "by requiring multiple detectors to agree."
        ),
        strengths=["Most robust — compensates individual weaknesses", "Semi-supervised mode with partial labels", "Produces well-calibrated unified score"],
        weaknesses=["Slower (trains multiple models)", "Harder to interpret which model contributed", "Overkill for simple univariate Gaussian data"],
        scores={
            "structure":      {"univariate": 0, "multivariate": 3, "time_series": 1},
            "distribution":   {"gaussian": 2, "skewed": 2, "heavy_tailed": 2, "multimodal": 2, "uniform": 1, "categorical": -1, "sparse": 1, "unknown": 3},
            "pattern":        {"stationary": 2, "trending": 1, "seasonal": 1, "random_walk": 0, "bursty": 2, "correlated": 2, "none": 2},
            "dimensionality": {"low": 0, "medium": 3, "high": 2},
            "anomaly_type":   {"point": 2, "contextual": 2, "collective": 1, "novelty": 2},
            "labels":         {"none": 2, "few": 3},
        },
        params={"threshold_percentile": 95},
    ),
    DetectorSpec(
        id="rolling_zscore",
        name="Rolling Z-Score",
        short="Sliding-window z-score for stationary time series",
        description=(
            "Computes z-scores within a sliding window, making it adaptive to "
            "slowly changing baselines. Flags points that spike far from their "
            "local neighbourhood mean. Simple and fast for univariate series."
        ),
        strengths=["Adaptive to local baseline", "Handles slow drift", "Very fast", "Interpretable"],
        weaknesses=["Univariate only", "Misses seasonal patterns", "Window size needs tuning"],
        time_series_only=True,
        scores={
            "structure":      {"univariate": 3, "multivariate": -1, "time_series": 3},
            "distribution":   {"gaussian": 3, "skewed": 0, "heavy_tailed": -1, "multimodal": -1, "uniform": 1, "categorical": -2, "sparse": 1, "unknown": 1},
            "pattern":        {"stationary": 3, "trending": 2, "seasonal": 0, "random_walk": 1, "bursty": 3, "correlated": -1, "none": 1},
            "dimensionality": {"low": 3, "medium": -1, "high": -2},
            "anomaly_type":   {"point": 3, "contextual": 2, "collective": 0, "novelty": 0},
            "labels":         {"none": 3, "few": 0},
        },
        params={"window": 20, "threshold": 3.0},
    ),
    DetectorSpec(
        id="stl",
        name="STL Decomposition",
        short="Trend + seasonality decomposition for periodic time series",
        description=(
            "Decomposes the series into trend, seasonal, and residual components "
            "using LOESS smoothing. Anomalies are points with unusually large "
            "residuals after accounting for trend and seasonality."
        ),
        strengths=["Handles trend AND seasonality simultaneously", "Robust variant available", "Anomaly score is interpretable residual"],
        weaknesses=["Requires specifying the period", "Univariate only", "Needs reasonably long series (≥2 full periods)"],
        time_series_only=True,
        scores={
            "structure":      {"univariate": 3, "multivariate": -2, "time_series": 3},
            "distribution":   {"gaussian": 2, "skewed": 1, "heavy_tailed": 0, "multimodal": 0, "uniform": 1, "categorical": -2, "sparse": 0, "unknown": 1},
            "pattern":        {"stationary": 1, "trending": 3, "seasonal": 3, "random_walk": 0, "bursty": 2, "correlated": -1, "none": 0},
            "dimensionality": {"low": 3, "medium": -1, "high": -3},
            "anomaly_type":   {"point": 3, "contextual": 3, "collective": 1, "novelty": 0},
            "labels":         {"none": 3, "few": 0},
        },
        params={"period": 12, "threshold_percentile": 95},
    ),
]

DETECTOR_MAP = {d.id: d for d in DETECTORS}


# ── Recommendation engine ──────────────────────────────────────────────────────

def recommend(selections: dict[str, list[str] | str], top_n: int = 5) -> list[tuple[DetectorSpec, int]]:
    """
    Returns detectors ranked by suitability score for the given nature selections.
    selections: {category: value_or_list_of_values}
    """
    is_time_series = selections.get("structure") in ("time_series", ["time_series"])

    scored = []
    for det in DETECTORS:
        if det.time_series_only and not is_time_series:
            continue  # hide TS-only methods for non-TS data

        total = 0
        for category, selected in selections.items():
            if not selected:
                continue
            cat_scores = det.scores.get(category, {})
            if isinstance(selected, str):
                selected = [selected]
            for val in selected:
                total += cat_scores.get(val, 0)

        scored.append((det, total))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


# ── Auto-detection helpers ─────────────────────────────────────────────────────

def auto_detect_natures(df) -> dict[str, list[str] | str]:
    """Heuristically infer data natures from a DataFrame."""
    import numpy as np
    from scipy import stats

    suggestions: dict[str, list[str] | str] = {}
    num_cols = df.select_dtypes(include="number").columns.tolist()
    n_cols = len(num_cols)
    n_rows = len(df)

    # Structure
    if n_cols == 1:
        suggestions["structure"] = "univariate"
    else:
        suggestions["structure"] = "multivariate"

    # Dimensionality
    if n_cols <= 5:
        suggestions["dimensionality"] = "low"
    elif n_cols <= 20:
        suggestions["dimensionality"] = "medium"
    else:
        suggestions["dimensionality"] = "high"

    # Distribution — test first numeric column
    dist_flags = []
    if num_cols and n_rows >= 8:
        col = df[num_cols[0]].dropna().values
        _, p_normal = stats.normaltest(col) if len(col) >= 8 else (None, 1.0)
        skewness = float(stats.skew(col))
        kurt = float(stats.kurtosis(col))

        if p_normal > 0.05:
            dist_flags.append("gaussian")
        if abs(skewness) > 1.0:
            dist_flags.append("skewed")
        if kurt > 3:
            dist_flags.append("heavy_tailed")

    # Check for categorical columns
    cat_cols = df.select_dtypes(exclude="number").columns
    if len(cat_cols) > 0 or any(df[c].nunique() < 20 for c in num_cols):
        dist_flags.append("categorical")

    # Sparsity
    if num_cols:
        zero_frac = (df[num_cols] == 0).values.mean()
        if zero_frac > 0.5:
            dist_flags.append("sparse")

    suggestions["distribution"] = dist_flags if dist_flags else ["unknown"]

    # Correlation
    pattern_flags = []
    if len(num_cols) >= 2 and n_rows >= 10:
        corr = df[num_cols].corr().abs()
        upper = corr.where(~corr.apply(lambda c: c.index <= c.name, axis=0))
        if (upper > 0.6).any().any():
            pattern_flags.append("correlated")

    # Trend (linear fit on index vs first numeric col)
    if num_cols and n_rows >= 10:
        col = df[num_cols[0]].dropna().values
        x = np.arange(len(col))
        slope, _, r, _, _ = stats.linregress(x, col)
        if abs(r) > 0.6:
            pattern_flags.append("trending")
        # Stationarity proxy: rolling std stable?
        rolling_std = (
            df[num_cols[0]].rolling(max(5, n_rows // 10)).std().dropna().std()
        )
        if rolling_std < df[num_cols[0]].std() * 0.3:
            pattern_flags.append("stationary")

    suggestions["pattern"] = pattern_flags if pattern_flags else ["none"]
    suggestions["anomaly_type"] = ["point"]
    suggestions["labels"] = "none"

    return suggestions
