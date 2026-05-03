from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_csv(path: str | Path, target_col: str | None = None) -> tuple[pd.DataFrame, pd.Series | None]:
    df = pd.read_csv(path)
    y = df.pop(target_col) if target_col and target_col in df.columns else None
    return df, y


def generate_synthetic(
    n_samples: int = 1000,
    n_features: int = 5,
    contamination: float = 0.05,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(random_state)
    n_anomalies = int(n_samples * contamination)
    n_normal = n_samples - n_anomalies

    normal = rng.standard_normal((n_normal, n_features))
    anomalies = rng.uniform(-6, 6, (n_anomalies, n_features))

    X = np.vstack([normal, anomalies])
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anomalies)]).astype(int)

    shuffle_idx = rng.permutation(n_samples)
    X, y = X[shuffle_idx], y[shuffle_idx]

    cols = [f"feature_{i}" for i in range(n_features)]
    return pd.DataFrame(X, columns=cols), y


def preprocess(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    return X_train_scaled, X_test_scaled, scaler
