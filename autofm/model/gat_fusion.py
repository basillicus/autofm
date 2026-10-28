"""GAT fusion stage: enrich per-window latents with cell-neighbour context.

Sits optionally between compression and the temporal classifier. It does NOT
change the classifier's input contract: in and out are both ``(N, k, latent)``.

How it preserves the contract:

1. Summarize each window to a single vector (mean over the ``k`` steps), then
   average those per **cell** -> one node feature per cell.
2. Build a topology adjacency over the cells present: cells sharing a gNB are
   fully connected (siblings under the same base station); a self-loop is always
   added. **The gNB a cell belongs to is taken from the dataset topology** (the
   ``cell -> gNB`` map derived from the ``fdn`` by the loader) when supplied -
   NOT guessed from the cell-id string. A regex suffix-strip fallback exists
   only for callers that cannot supply the map, and is lossy (see ``gnb_of_cell``).
3. Run :class:`~autofm.model.gat.TopologyGAT`; project the per-cell enrichment
   back to ``latent`` with a linear layer and **add** it (broadcast across the
   window's time steps) to every window of that cell.

**What is and is not trained here.** This stage is *not* optimised against the
task loss: the GAT and the back-projection run in eager forward with weights
fixed at construction by ``seed``. It therefore contributes a **deterministic,
seed-fixed, topology-aware mixing of neighbour features** (real signal - a cell
is enriched with its gNB-siblings' representation), and the returned attention
is a readable topology attention, not a task-learned saliency. Because the
weights are seed-fixed, reusing the *same* ``GATFusion`` instance (same process,
same device) across train then val/test yields a consistent transform, so the
classifier sees at eval the same representation family it saw at train. Callers
MUST reuse one instance across splits rather than constructing a fresh one per
split. (Weight init is not guaranteed bitwise-identical across CPU/MPS/CUDA, so
reuse the instance rather than relying on re-seeding a new one.)

No label leakage: enrichment uses only features + topology, never labels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from autofm.model.gat import TopologyGAT

# Fallback only: gNB key ~= cell id without the trailing sector index + letter
# (e.g. BAJE1U -> BAJE). This is a LOSSY heuristic used only when no real
# cell->gNB topology map is supplied; prefer the map (see GATFusion).
_SECTOR_SUFFIX = re.compile(r"\d+[A-Za-z]?$")


def gnb_of_cell(cell: str) -> str:
    """Heuristic gNB key from a cell id by stripping the trailing sector suffix.

    FALLBACK ONLY. The authoritative gNB for a cell comes from the ``fdn``
    (the loader's ``MeContext``-derived ``gnb`` column), which does not in
    general equal any substring of the cell id. Use the ``cell_to_gnb`` map on
    :class:`GATFusion` whenever it is available.
    """
    return _SECTOR_SUFFIX.sub("", str(cell)) or str(cell)


def _cell_index(cells: np.ndarray) -> tuple[list[str], dict[str, int]]:
    """Stable ordered unique cell ids + a lookup from cell id to node index."""
    uniq = sorted(set(map(str, cells.tolist())))
    return uniq, {c: i for i, c in enumerate(uniq)}


def _node_features(Z: np.ndarray, cells: np.ndarray,
                   cell_ids: list[str], idx: dict[str, int],
                   latent: int, device, node_of_row: np.ndarray) -> torch.Tensor:
    """Mean-pool windows over time, then average per cell -> (C, latent).

    Vectorized: ``node_of_row`` maps each window row to its node index; the
    per-node sum/count are accumulated with ``np.add.at`` (no Python loop over
    the N windows).
    """
    per_window = Z.mean(axis=1).astype(np.float32)   # (N, latent)
    sums = np.zeros((len(cell_ids), latent), dtype=np.float32)
    counts = np.zeros((len(cell_ids), 1), dtype=np.float32)
    np.add.at(sums, node_of_row, per_window)
    np.add.at(counts, node_of_row, 1.0)
    np.maximum(counts, 1.0, out=counts)              # guard empty nodes
    return torch.from_numpy(sums / counts).to(device)


@dataclass
class GATFusion:
    """Builds topology, runs the GAT, and enriches compressed windows.

    Args:
        latent_dim: per-window latent size (in == out, contract preserved).
        cell_to_gnb: authoritative ``cell id -> gNB`` map from the dataset
            topology (loader ``fdn`` parse). When provided, cells are grouped by
            their real gNB; when ``None``, a lossy regex fallback is used and a
            one-time warning is printed.
        hidden/out_dim/heads: GAT widths.
        seed: fixes the (untrained) GAT + projection weights for determinism.
    """

    latent_dim: int
    cell_to_gnb: dict[str, str] | None = None
    hidden: int = 64
    out_dim: int = 64
    heads: int = 4
    seed: int = 42
    _adj_cache: dict[tuple[str, ...], torch.Tensor] = field(
        default_factory=dict, init=False, repr=False)

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
        self._warned_fallback = False
        self._missing_cells: set[str] = set()

    def _gnb_of(self, cell: str) -> str:
        """Real gNB from the topology map, or the lossy regex fallback.

        Warns (once) whenever the fallback is used - whether because no map was
        supplied at all, or because a specific cell is missing from the map -
        so a silently-wrong grouping cannot pass unnoticed.
        """
        cell = str(cell)
        if self.cell_to_gnb is not None:
            gnb = self.cell_to_gnb.get(cell)
            if gnb is not None:
                return str(gnb)
            self._missing_cells.add(cell)
        if not self._warned_fallback:
            reason = ("no cell->gNB topology map supplied"
                      if self.cell_to_gnb is None
                      else "some cells are missing from the cell->gNB map")
            print(f"[gat] WARNING: {reason}; falling back to lossy cell-id "
                  f"suffix strip for grouping (groups may be wrong)")
            self._warned_fallback = True
        return gnb_of_cell(cell)

    def _within_gnb_adjacency(self, cell_ids: list[str]) -> torch.Tensor:
        """Dense (C, C) adjacency (1 within a gNB, plus self-loops), cached.

        The adjacency depends only on the set of cell ids (and the topology
        map), so it is built once per unique cell set and reused across splits
        and epochs rather than rebuilt on every enrich call.
        """
        key = tuple(cell_ids)
        cached = self._adj_cache.get(key)
        if cached is not None:
            return cached

        n = len(cell_ids)
        adj = torch.eye(n, device=self.device)
        # Group node indices by real gNB, then set the within-group block to 1
        # via vectorized index_put_ (no Python triple loop).
        groups: dict[str, list[int]] = {}
        for i, c in enumerate(cell_ids):
            groups.setdefault(self._gnb_of(c), []).append(i)
        for members in groups.values():
            if len(members) <= 1:
                continue
            m = torch.tensor(members, device=self.device)
            rows = m.repeat_interleave(len(m))
            cols = m.repeat(len(m))
            adj[rows, cols] = 1.0

        self._adj_cache[key] = adj
        return adj

    def _enrich(self, Z: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """Return Z with per-cell GAT neighbour context added. Shape preserved."""
        cell_ids, idx = _cell_index(cells)
        adj = self._within_gnb_adjacency(cell_ids)
        # Map every window row to its node index once (shared by the scatter and
        # the broadcast-back), so neither does a Python loop over N windows.
        node_of_row = np.fromiter((idx[str(c)] for c in cells), dtype=np.int64,
                                  count=len(cells))
        x = _node_features(Z, cells, cell_ids, idx, self.latent_dim, self.device,
                           node_of_row=node_of_row)

        node_emb, attention = self.gat(x, adj)       # (C, out), (C, C)
        enrich = self.project(node_emb)              # (C, latent)

        self.last_attention = attention.detach()
        self.last_cell_ids = cell_ids

        # Broadcast each cell's enrichment onto all its windows + time steps,
        # vectorized: gather the per-node enrichment by row, add across the
        # window's k time steps at once (no Python loop over N windows).
        enrich_np = enrich.detach().cpu().numpy()    # (C, latent)
        out = Z.astype(np.float32, copy=True)
        out += enrich_np[node_of_row][:, None, :]    # (N,1,latent) broadcast over k
        return out

    def fit_transform(self, Z: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """Fit on the training split (records its cells) and enrich it."""
        self._fitted_cells = sorted(set(map(str, cells.tolist())))
        return self._enrich(Z, cells)

    def transform(self, Z: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """Enrich a val/test split with the (seed-fixed) GAT; shape preserved."""
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
