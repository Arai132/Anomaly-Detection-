import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .base import BaseDetector


class _VAENetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


class VAEDetector(BaseDetector):
    """
    PyTorch variational autoencoder. Trained with reconstruction loss +
    beta-weighted KL divergence against a standard normal prior.
    Anomaly score = reconstruction error from the mean latent code `mu`
    (deterministic, no sampling) so scoring is repeatable at inference time.
    Threshold set at `threshold_percentile` of training scores.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int | None = None,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        beta: float = 1.0,
        threshold_percentile: float = 95,
        random_state: int = 42,
        device: str = "auto",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.beta = beta
        self.threshold_percentile = threshold_percentile
        self.random_state = random_state
        self.device = device
        self._threshold: float | None = None

    def _resolve_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X: pd.DataFrame | np.ndarray, y=None) -> "VAEDetector":
        arr = self._to_array(X).astype(np.float32)
        torch.manual_seed(self.random_state)

        n_features = arr.shape[1]
        hidden_dim = self.hidden_dim or max(2 * self.latent_dim, n_features)
        self._device = self._resolve_device()
        self._model = _VAENetwork(n_features, hidden_dim, self.latent_dim).to(self._device)

        loader = DataLoader(
            TensorDataset(torch.from_numpy(arr)),
            batch_size=min(self.batch_size, len(arr)),
            shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state),
        )
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        self._model.train()
        for _ in range(self.epochs):
            for (batch,) in loader:
                batch = batch.to(self._device)
                optimizer.zero_grad()
                recon, mu, logvar = self._model(batch)
                recon_loss = torch.mean((recon - batch) ** 2, dim=1)
                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                loss = (recon_loss + self.beta * kld).mean()
                loss.backward()
                optimizer.step()

        scores = self._reconstruction_error(arr)
        self._threshold = float(np.percentile(scores, self.threshold_percentile))
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return (self.score_samples(X) > self._threshold).astype(int)

    def score_samples(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        arr = self._to_array(X).astype(np.float32)
        return self._reconstruction_error(arr)

    def _reconstruction_error(self, arr: np.ndarray) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            x = torch.from_numpy(arr).to(self._device)
            mu, _ = self._model.encode(x)
            recon = self._model.decoder(mu)
            mse = torch.mean((recon - x) ** 2, dim=1)
        return mse.cpu().numpy()

    def save(self, path) -> None:
        trained_device = self._device
        self._model.to("cpu")
        self._device = torch.device("cpu")
        try:
            super().save(path)
        finally:
            self._model.to(trained_device)
            self._device = trained_device

    @classmethod
    def load(cls, path) -> "VAEDetector":
        obj = super().load(path)
        obj._device = obj._resolve_device()
        obj._model.to(obj._device)
        return obj
