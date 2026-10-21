"""Temporal fault classifier (PatchTST-lite).

Takes the compressed window ``(N, k, latent)`` and predicts the fault class at
the forecast horizon. A light Transformer encoder over the ``k`` time steps
(the autofm design uses PatchTST; this is a compact, GPU-friendly stand-in)
feeds a linear classification head.

Class imbalance (most windows are Normal) is handled with class-weighted
cross-entropy. The model is device-aware (CUDA > MPS > CPU) so the same code
runs on a laptop and on Leonardo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _TemporalEncoder(nn.Module):
    """Linear patch/embed -> Transformer encoder -> mean-pool -> class logits."""

    def __init__(self, latent_dim: int, n_classes: int,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(latent_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, k, L) -> (B, C)
        h = self.embed(x)
        h = self.encoder(h)
        h = h.mean(dim=1)       # pool over time
        return self.head(h)


@dataclass
class TemporalClassifier:
    """Trains the temporal encoder and reports metrics."""

    latent_dim: int
    n_classes: int
    epochs: int = 12
    batch_size: int = 256
    lr: float = 1e-3
    seed: int = 42
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    _net: _TemporalEncoder | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        torch.manual_seed(self.seed)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu")
        self._net = _TemporalEncoder(
            self.latent_dim, self.n_classes,
            d_model=self.d_model, n_heads=self.n_heads, n_layers=self.n_layers,
        ).to(self.device)

    def _class_weights(self, y: np.ndarray) -> torch.Tensor:
        """Inverse-frequency weights so minority fault classes are not ignored."""
        counts = np.bincount(y, minlength=self.n_classes).astype(np.float64)
        counts[counts == 0] = 1.0
        w = counts.sum() / (self.n_classes * counts)
        return torch.tensor(w, dtype=torch.float32, device=self.device)

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> dict:
        assert self._net is not None
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X.astype(np.float32)),
                          torch.from_numpy(y.astype(np.int64))),
            batch_size=self.batch_size, shuffle=True,
        )
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss(weight=self._class_weights(y))

        history = {"train_loss": [], "val_macro_f1": []}
        for epoch in range(1, self.epochs + 1):
            self._net.train()
            total = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self._net(xb)
                loss = loss_fn(logits, yb)
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item() * len(xb)
            total /= len(X)
            history["train_loss"].append(total)

            vf1 = (self.macro_f1(X_val, y_val)
                   if X_val is not None and len(X_val) else float("nan"))
            history["val_macro_f1"].append(vf1)
            print(f"  [clf] epoch {epoch:02d}/{self.epochs} "
                  f"loss={total:.4f} val_macroF1={vf1:.4f}")
        return history

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._net is not None
        self._net.eval()
        out = []
        for i in range(0, len(X), self.batch_size):
            xb = torch.from_numpy(X[i:i + self.batch_size].astype(np.float32)).to(self.device)
            out.append(self._net(xb).argmax(dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.empty(0, dtype=np.int64)

    def macro_f1(self, X: np.ndarray, y: np.ndarray) -> float:
        from sklearn.metrics import f1_score
        pred = self.predict(X)
        return float(f1_score(y, pred, average="macro", zero_division=0))
