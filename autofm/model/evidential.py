"""Evidential Deep Learning head (Dirichlet) for fault classification.

A plain softmax classifier gives a class but no honest "how sure am I". EDL
(Sensoy et al., 2018) treats the network output as **evidence** for a Dirichlet
distribution over the class probabilities, which yields both a prediction and a
calibrated **uncertainty** in one shot.

Mechanics (K = number of classes):

* The net emits non-negative evidence ``e >= 0`` (softplus of the logits).
* Dirichlet parameters ``alpha = e + 1``; Dirichlet strength ``S = sum(alpha)``.
* Expected class probabilities ``p = alpha / S``.
* **Uncertainty (vacuity)** ``u = K / S`` in ``[0, 1]`` - high when total
  evidence is low (the model has seen little support for any class).

Loss = Bayes-risk of the expected sum-of-squares error (the ``edl_mse`` form,
which is stable to train) + a KL regularizer that pulls the Dirichlet of
*misleading* evidence back toward uniform. The KL term is annealed in over the
first several epochs so early training is not dominated by regularization.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class EvidentialHead(nn.Module):
    """Maps a feature vector to non-negative Dirichlet evidence.

    Softplus keeps evidence >= 0; ``alpha = evidence + 1`` is formed by the loss
    and inference helpers, not here, so this stays a thin linear head.
    """

    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # -> evidence (B, K)
        return F.softplus(self.fc(self.norm(x)))


def evidence_to_alpha(evidence: torch.Tensor) -> torch.Tensor:
    """Dirichlet concentration parameters alpha = evidence + 1."""
    return evidence + 1.0


def dirichlet_uncertainty(alpha: torch.Tensor) -> torch.Tensor:
    """Vacuity uncertainty u = K / S per sample, in [0, 1]."""
    k = alpha.shape[-1]
    strength = alpha.sum(dim=-1)
    return k / strength


def dirichlet_probs(alpha: torch.Tensor) -> torch.Tensor:
    """Expected class probabilities p = alpha / S."""
    return alpha / alpha.sum(dim=-1, keepdim=True)


def _kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """KL( Dir(alpha) || Dir(1) ) - penalizes evidence on the wrong classes.

    Standard closed form of the KL divergence between a Dirichlet and the
    uniform Dirichlet (all-ones), computed per sample.
    """
    k = alpha.shape[-1]
    ones = torch.ones_like(alpha)
    s_alpha = alpha.sum(dim=-1, keepdim=True)
    s_ones = ones.sum(dim=-1, keepdim=True)

    t1 = torch.lgamma(s_alpha) - torch.lgamma(s_ones)
    t2 = (torch.lgamma(ones).sum(dim=-1, keepdim=True)
          - torch.lgamma(alpha).sum(dim=-1, keepdim=True))
    t3 = ((alpha - ones)
          * (torch.digamma(alpha) - torch.digamma(s_alpha))).sum(dim=-1, keepdim=True)
    return (t1 + t2 + t3).squeeze(-1)


def edl_mse_loss(
    evidence: torch.Tensor,
    targets: torch.Tensor,
    *,
    n_classes: int,
    anneal: float = 1.0,
    class_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evidential MSE (Bayes-risk) loss + annealed KL regularizer.

    Args:
        evidence:     (B, K) non-negative evidence from :class:`EvidentialHead`.
        targets:      (B,) integer class ids.
        n_classes:    K.
        anneal:       KL weight in [0, 1], ramped from 0 -> 1 over early epochs.
        class_weight: optional (K,) per-class weights for imbalance (applied to
                      the per-sample loss by the sample's true class).

    Returns a scalar loss (mean over the batch, class-weighted if given).
    """
    alpha = evidence_to_alpha(evidence)
    strength = alpha.sum(dim=-1, keepdim=True)
    p = alpha / strength
    y = F.one_hot(targets, num_classes=n_classes).float()

    # Bayes risk of the expected sum-of-squares error: a data-fit term plus a
    # variance term, both in closed form for a Dirichlet.
    err = ((y - p) ** 2).sum(dim=-1)
    var = (p * (1.0 - p) / (strength + 1.0)).sum(dim=-1)
    data_term = err + var

    # KL regularizer on the evidence of the *wrong* classes only: remove the
    # true-class evidence, then pull the remainder toward uniform.
    alpha_tilde = y + (1.0 - y) * alpha
    kl = _kl_to_uniform(alpha_tilde)

    per_sample = data_term + anneal * kl

    if class_weight is not None:
        w = class_weight[targets]
        return (per_sample * w).sum() / w.sum()
    return per_sample.mean()
