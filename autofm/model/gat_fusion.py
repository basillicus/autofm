"""GAT fusion stage: enrich per-window latents with cell-neighbour context.

Sits optionally between compression and the temporal classifier. It does NOT
change the classifier's input contract: in and out are both ``(N, k, latent)``.

How it preserves the contract:

1. Summarize each window to a single vector (mean over the ``k`` steps), then
   average those per **cell** -> one node feature per cell.
2. Build a topology adjacency over the cells present. With no external
   topology, cells sharing a gNB are fully connected (siblings under the same
   base station); a self-loop is always added. The gNB key is the cell id with
   its trailing sector digit stripped (``BAJE1U`` / ``BAJE2U`` -> ``BAJE``).
3. Run :class:`~autofm.model.gat.TopologyGAT`; project the per-cell enrichment
   back to ``latent`` with a linear layer and **add** it (broadcast across the
   window's time steps) to every window of that cell.

The GAT's attention matrix is returned so a causal chain can be traced for the
predicted-fault example (feeds the advisory). Fit the GAT on the training
windows, then apply the same enrichment to val/test (no label leakage: it uses
only features + topology).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from autofm.model.gat import TopologyGAT

# gNB key = cell id without the trailing sector index + 'U' (e.g. BAJE1U->BAJE).
_SECTOR_SUFFIX = re.compile(r"\d+[A-Za-z]?$")


def gnb_of_cell(cell: str) -> str:
    """Strip the trailing sector suffix to get the owning gNB key."""
    return _SECTOR_SUFFIX.sub("", str(cell)) or str(cell)


def _cell_index(cells: np.ndarray) -> tuple[list[str], dict[str, int]]:
    """Stable ordered unique cell ids + a lookup from cell id to node index."""
    uniq = sorted(set(map(str, cells.tolist())))
    return uniq, {c: i for i, c in enumerate(uniq)}


def _within_gnb_adjacency(cell_ids: list[str], device) -> torch.Tensor:
    """Dense (C, C) adjacency: 1 between cells of the same gNB, plus self-loops."""
    n = len(cell_ids)
    adj = torch.eye(n, device=device)
    groups: dict[str, list[int]] = {}
    for i, c in enumerate(cell_ids):
        groups.setdefault(gnb_of_cell(c), []).append(i)
    for members in groups.values():
        for i in members:
            for j in members:
                adj[i, j] = 1.0
    return adj


def _node_features(Z: np.ndarray, cells: np.ndarray,
                   cell_ids: list[str], idx: dict[str, int],
                   latent: int, device) -> torch.Tensor:
    """Mean-pool windows over time, then average per cell -> (C, latent)."""
    per_window = Z.mean(axis=1)                      # (N, latent)
    feats = torch.zeros(len(cell_ids), latent, device=device)
    counts = torch.zeros(len(cell_ids), 1, device=device)
    pw = torch.from_numpy(per_window.astype(np.float32)).to(device)
    for row, c in enumerate(cells):
        node = idx[str(c)]
        feats[node] += pw[row]
        counts[node] += 1
    counts = counts.clamp_min(1.0)
    return feats / counts


@dataclass
class GATFusion:
    """Builds topology, runs the GAT, and enriches compressed windows."""

    latent_dim: int
    hidden: int = 64
    out_dim: int = 64
    heads: int = 4
    seed: int = 42

    def __post_init__(self) -> None:
        torch.manual_seed(self.seed)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu")
        self.gat = TopologyGAT(self.latent_dim, hidden=self.hidden,
                               out_dim=self.out_dim, heads=self.heads).to(self.device)
        # Project GAT output back to latent so the classifier contract holds.
        self.project = nn.Linear(self.out_dim, self.latent_dim).to(self.device)
        self._fitted_cells: list[str] | None = None
        self.last_attention: torch.Tensor | None = None
        self.last_cell_ids: list[str] | None = None

    def _enrich(self, Z: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """Return Z with per-cell GAT neighbour context added. Shape preserved."""
        cell_ids, idx = _cell_index(cells)
        adj = _within_gnb_adjacency(cell_ids, self.device)
        x = _node_features(Z, cells, cell_ids, idx, self.latent_dim, self.device)

        node_emb, attention = self.gat(x, adj)       # (C, out), (C, C)
        enrich = self.project(node_emb)              # (C, latent)

        self.last_attention = attention.detach()
        self.last_cell_ids = cell_ids

        # Broadcast each cell's enrichment onto all its windows + time steps.
        enrich_np = enrich.detach().cpu().numpy()
        out = Z.copy()
        for row, c in enumerate(cells):
            out[row] += enrich_np[idx[str(c)]][None, :]   # (k, latent) += (latent,)
        return out.astype(np.float32)

    def fit_transform(self, Z: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """Fit on the training split (records its cells) and enrich it."""
        self._fitted_cells = sorted(set(map(str, cells.tolist())))
        return self._enrich(Z, cells)

    def transform(self, Z: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """Enrich a val/test split with the (eager) GAT; shape preserved."""
        if len(Z) == 0:
            return Z
        return self._enrich(Z, cells)

    def causal_chain(self, node_cell: str, depth: int = 3) -> list[str]:
        """Trace the highest-attention propagation chain from a cell.

        Returns a list of **cell ids** (not indices) so it is human-readable in
        the advisory. Empty if no attention has been computed yet.
        """
        if self.last_attention is None or self.last_cell_ids is None:
            return []
        if str(node_cell) not in self.last_cell_ids:
            return []
        start = self.last_cell_ids.index(str(node_cell))
        chain_idx = TopologyGAT.causal_chain(self.last_attention, start, depth=depth)
        return [self.last_cell_ids[i] for i in chain_idx]
