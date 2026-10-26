"""PatchTST backbone - patch-based Transformer over the time window.

A drop-in alternative to the plain per-step Transformer in
:mod:`autofm.model.classifier`. Instead of embedding every one of the ``k``
time steps, PatchTST (Nie et al., 2023) slices the window into fixed-length
**patches** and embeds each patch, so the Transformer attends over far fewer
tokens (``k / patch_size``). That makes it cheaper and lets each token carry
local temporal context.

Shapes (B = batch, k = window length, L = latent/feature dim per step):

    input   (B, k, L)
      -> fold into patches        (B, n_patches, patch_size * L)
      -> linear patch embedding   (B, n_patches, d_model)
      -> + learned positional emb
      -> Transformer encoder      (B, n_patches, d_model)
      -> mean-pool over patches   (B, d_model)

The default is ``patch_size=5`` over a 60-step window -> 12 patches. When the
window length is not divisible by the patch size, the sequence is left-padded
with zeros to the next multiple, so any ``(k, patch_size)`` runs. Pure PyTorch
(``nn.TransformerEncoder``), so it runs unchanged on CPU, Apple MPS, and CUDA.

Exposes the same surface the classifier expects - ``features(x)`` returning a
pooled ``(B, d_model)`` vector - so softmax or evidential heads sit on top
without change.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PatchTSTBackbone(nn.Module):
    """Patch-embed -> Transformer encoder -> pooled window feature."""

    def __init__(
        self,
        latent_dim: int,
        *,
        window_k: int = 60,
        patch_size: int = 5,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {patch_size}")
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.d_model = d_model

        # Pad the window up to a whole number of patches if needed.
        self._pad = (-window_k) % patch_size      # 0 when divisible
        padded_k = window_k + self._pad
        self.n_patches = padded_k // patch_size

        self.patch_embed = nn.Linear(patch_size * latent_dim, d_model)
        # Learned positional embedding over the patch tokens.
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def features(self, x: torch.Tensor) -> torch.Tensor:  # (B, k, L) -> (B, d)
        b, k, lat = x.shape
        if self._pad:
            # Left-pad time with zeros so k becomes a multiple of patch_size.
            x = F.pad(x, (0, 0, self._pad, 0))
        # Fold the time axis into patches: (B, n_patches, patch_size * L).
        patches = x.reshape(b, self.n_patches, self.patch_size * lat)
        h = self.patch_embed(patches) + self.pos
        h = self.encoder(h)
        return self.norm(h.mean(dim=1))       # pool over patch tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
