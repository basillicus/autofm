"""Auxiliary prediction heads: time-to-failure and localization.

Beyond "what fault", an operator needs **when** and **where**. These two light
heads sit on the shared window feature (the ``d_model`` vector from the
temporal backbone) and are trained alongside the classifier.

* :class:`TTFHead` - a **quantile** regressor predicting time-to-failure at the
  0.1 / 0.5 / 0.9 quantiles, trained with the pinball (quantile) loss. The 0.5
  output is the median ETA; the 0.1-0.9 pair is an 80% interval. TTF is only
  defined for degrading samples, so the loss is masked to those rows.

* :class:`LocalizationHead` - a per-node classifier over the topology nodes
  (host / container / cell / ...), giving an origin-node ranking via softmax.

Both are plain ``nn.Module`` and run on CPU / MPS / CUDA.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TTF_QUANTILES = (0.1, 0.5, 0.9)


class TTFHead(nn.Module):
    """Quantile time-to-failure head: feature -> (q0.1, q0.5, q0.9)."""

    def __init__(self, in_dim: int, quantiles: tuple[float, ...] = TTF_QUANTILES):
        super().__init__()
        self.quantiles = quantiles
        self.norm = nn.LayerNorm(in_dim)
        self.fc = nn.Linear(in_dim, len(quantiles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # -> (B, n_quantiles)
        # Softplus keeps TTF non-negative (you can't fail in negative time).
        return F.softplus(self.fc(self.norm(x)))

    def predict_interval(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return median + lower/upper bound tensors for a batch."""
        out = self.forward(x)
        qi = {q: out[:, i] for i, q in enumerate(self.quantiles)}
        return {
            "median": qi.get(0.5, out[:, out.shape[1] // 2]),
            "lower": qi.get(0.1, out[:, 0]),
            "upper": qi.get(0.9, out[:, -1]),
        }


def pinball_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantiles: tuple[float, ...] = TTF_QUANTILES,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Quantile (pinball) loss, averaged over quantiles and valid rows.

    Args:
        pred:      (B, Q) predicted quantiles.
        target:    (B,) regression targets (time-to-failure in ROPs).
        quantiles: the Q quantile levels matching ``pred`` columns.
        mask:      optional (B,) bool/0-1 mask selecting degrading rows; when
                   given, non-degrading rows contribute no loss (TTF is only
                   defined during a degradation ramp).
    """
    target = target.unsqueeze(1)                 # (B, 1)
    errors = target - pred                       # (B, Q)
    q = torch.tensor(quantiles, dtype=pred.dtype, device=pred.device)
    # pinball: max(q*e, (q-1)*e) per element.
    loss = torch.maximum(q * errors, (q - 1.0) * errors)   # (B, Q)
    loss = loss.mean(dim=1)                                 # (B,)
    if mask is not None:
        m = mask.to(loss.dtype)
        denom = m.sum().clamp_min(1.0)
        return (loss * m).sum() / denom
    return loss.mean()


class LocalizationHead(nn.Module):
    """Per-node localization: feature -> logits over topology nodes."""

    def __init__(self, in_dim: int, n_nodes: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc = nn.Linear(in_dim, n_nodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # -> (B, n_nodes)
        return self.fc(self.norm(x))

    def rank(self, x: torch.Tensor, k: int = 3) -> torch.Tensor:
        """Top-k most likely origin nodes per sample (indices)."""
        logits = self.forward(x)
        k = min(k, logits.shape[1])
        return logits.topk(k, dim=1).indices


def ttf_report(head: "TTFHead", feats: torch.Tensor) -> dict[str, float]:
    """Summary stats: median TTF and the mean 80% interval, over a batch."""
    with torch.no_grad():
        iv = head.predict_interval(feats)
        return {
            "ttf_median": float(iv["median"].mean().item()),
            "ttf_lower_p10": float(iv["lower"].mean().item()),
            "ttf_upper_p90": float(iv["upper"].mean().item()),
        }
