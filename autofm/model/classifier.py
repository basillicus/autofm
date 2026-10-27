"""Temporal fault classifier (PatchTST-lite).

Takes the compressed window ``(N, k, latent)`` and predicts the fault class at
the forecast horizon. A light Transformer encoder over the ``k`` time steps
(the autofm design uses PatchTST; this is a compact, GPU-friendly stand-in)
feeds a classification head.

Two heads, selectable via ``head``:

* ``"softmax"`` (default) - plain class-weighted cross-entropy. The baseline.
* ``"evidential"`` - a Dirichlet Evidential head (:mod:`autofm.model.evidential`)
  that also yields a per-prediction **uncertainty**. Trained with the
  evidential MSE + annealed KL loss.

Both report macro-F1; the evidential head additionally reports average
uncertainty. Class imbalance (most windows are Normal) is handled with
inverse-frequency class weights. Device-aware (CUDA > MPS > CPU).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from autofm.model.evidential import (
    EvidentialHead,
    dirichlet_probs,
    dirichlet_uncertainty,
    edl_mse_loss,
    evidence_to_alpha,
)
from autofm.model.patchtst import PatchTSTBackbone


class _TemporalEncoder(nn.Module):
    """Linear embed -> Transformer encoder -> mean-pool over time.

    Returns a feature vector by default; with ``head="softmax"`` it appends a
    linear classifier so the plain path is self-contained. The evidential path
    uses the features and attaches its own head.
    """

    def __init__(self, latent_dim: int, n_classes: int,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.1, head: str = "softmax",
                 backbone: str = "transformer", window_k: int = 60,
                 patch_size: int = 5):
        super().__init__()
        self.head_kind = head
        self.backbone_kind = backbone

        if backbone == "transformer":
            # Per-step embed -> Transformer -> mean-pool over time.
            self.embed = nn.Linear(latent_dim, d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
                dropout=dropout, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            self._backbone = None
        elif backbone == "patchtst":
            # Patch-based Transformer (see autofm.model.patchtst).
            self._backbone = PatchTSTBackbone(
                latent_dim, window_k=window_k, patch_size=patch_size,
                d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone!r}")

        if head == "softmax":
            self.head = nn.Sequential(
                nn.LayerNorm(d_model), nn.Linear(d_model, n_classes),
            )
        elif head == "evidential":
            self.head = EvidentialHead(d_model, n_classes)
        else:
            raise ValueError(f"Unknown head: {head!r}")

    def features(self, x: torch.Tensor) -> torch.Tensor:  # (B, k, L) -> (B, d)
        if self._backbone is not None:
            return self._backbone.features(x)
        h = self.embed(x)
        h = self.encoder(h)
        return h.mean(dim=1)       # pool over time

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # softmax head -> logits; evidential head -> non-negative evidence.
        return self.head(self.features(x))


@dataclass
class TemporalClassifier:
    """Trains the temporal encoder and reports metrics.

    ``head`` selects the plain softmax classifier or the Dirichlet evidential
    head. ``anneal_epochs`` controls how fast the evidential KL term ramps in.
    """

    latent_dim: int
    n_classes: int
    epochs: int = 12
    batch_size: int = 256
    lr: float = 1e-3
    seed: int = 42
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    head: str = "softmax"          # "softmax" | "evidential"
    anneal_epochs: int = 10
    backbone: str = "transformer"  # "transformer" | "patchtst"
    window_k: int = 60             # used by the patchtst backbone
    patch_size: int = 5            # patch length for patchtst
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
            head=self.head, backbone=self.backbone,
            window_k=self.window_k, patch_size=self.patch_size,
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
        class_w = self._class_weights(y)
        ce = nn.CrossEntropyLoss(weight=class_w)

        history: dict = {"train_loss": [], "val_macro_f1": []}
        if self.head == "evidential":
            history["val_uncertainty"] = []

        for epoch in range(1, self.epochs + 1):
            self._net.train()
            # KL weight ramps 0 -> 1 across the first ``anneal_epochs`` epochs.
            anneal = min(1.0, epoch / max(self.anneal_epochs, 1))
            total = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                out = self._net(xb)
                if self.head == "evidential":
                    loss = edl_mse_loss(out, yb, n_classes=self.n_classes,
                                        anneal=anneal, class_weight=class_w)
                else:
                    loss = ce(out, yb)
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item() * len(xb)
            total /= len(X)
            history["train_loss"].append(total)

            vf1 = (self.macro_f1(X_val, y_val)
                   if X_val is not None and len(X_val) else float("nan"))
            history["val_macro_f1"].append(vf1)
            msg = (f"  [clf:{self.head}] epoch {epoch:02d}/{self.epochs} "
                   f"loss={total:.4f} val_macroF1={vf1:.4f}")
            if self.head == "evidential":
                vu = (self.mean_uncertainty(X_val)
                      if X_val is not None and len(X_val) else float("nan"))
                history["val_uncertainty"].append(vu)
                msg += f" val_uncertainty={vu:.4f}"
            print(msg)
        return history

    @torch.no_grad()
    def _forward_all(self, X: np.ndarray) -> torch.Tensor:
        """Run the net over X in batches, returning raw head outputs on CPU."""
        assert self._net is not None
        self._net.eval()
        outs = []
        for i in range(0, len(X), self.batch_size):
            xb = torch.from_numpy(X[i:i + self.batch_size].astype(np.float32)).to(self.device)
            outs.append(self._net(xb).cpu())
        return torch.cat(outs) if outs else torch.empty(0, self.n_classes)

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.empty(0, dtype=np.int64)
        out = self._forward_all(X)
        if self.head == "evidential":
            # argmax of alpha == argmax of evidence; either gives the class.
            alpha = evidence_to_alpha(out)
            return dirichlet_probs(alpha).argmax(dim=1).numpy()
        return out.argmax(dim=1).numpy()

    @torch.no_grad()
    def predict_with_uncertainty(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (class predictions, per-sample uncertainty in [0, 1]).

        Only meaningful for the evidential head; the softmax head reports
        ``1 - max p`` as a rough confidence-gap proxy so the API is uniform.
        """
        if len(X) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        out = self._forward_all(X)
        if self.head == "evidential":
            alpha = evidence_to_alpha(out)
            pred = dirichlet_probs(alpha).argmax(dim=1).numpy()
            unc = dirichlet_uncertainty(alpha).numpy().astype(np.float32)
        else:
            probs = torch.softmax(out, dim=1)
            pred = probs.argmax(dim=1).numpy()
            unc = (1.0 - probs.max(dim=1).values).numpy().astype(np.float32)
        return pred, unc

    def mean_uncertainty(self, X: np.ndarray) -> float:
        """Average predictive uncertainty over X."""
        if X is None or len(X) == 0:
            return float("nan")
        _, unc = self.predict_with_uncertainty(X)
        return float(unc.mean())

    def macro_f1(self, X: np.ndarray, y: np.ndarray) -> float:
        from sklearn.metrics import f1_score
        pred = self.predict(X)
        return float(f1_score(y, pred, average="macro", zero_division=0))
