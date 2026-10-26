"""Graph Attention Network over the gNB topology.

Fault symptoms propagate along the network graph (host -> container -> cell,
and between sibling/neighbour cells). A GAT lets each node refine its embedding
by attending to its neighbours, and - crucially - the learned **attention
weights are readable as a causal chain**: for a predicted fault node, the edges
it attended to most strongly are the likeliest propagation path.

Two interchangeable backends behind one interface:

* **PyG path** (preferred): ``torch_geometric.nn.GATConv`` with
  ``return_attention_weights`` so we get per-edge attention directly.
* **Fallback path** (no PyG): a plain-PyTorch masked attention over the dense
  adjacency matrix. Same idea, same outputs (node embeddings + an attention
  matrix), just without the PyG dependency - so the program never crashes if
  ``torch_geometric`` is absent.

The active path is printed once at construction so it is obvious which ran.
Both expose ``forward(x, adj/edge_index)`` returning
``(node_embeddings, attention)`` and a ``causal_chain(...)`` helper that traces
the highest-attention path back from a node.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

try:  # Preferred: PyTorch Geometric's optimized sparse GAT.
    from torch_geometric.nn import GATConv
    _HAS_PYG = True
except Exception:  # noqa: BLE001 - any import failure falls back gracefully
    _HAS_PYG = False


# --------------------------------------------------------------------------- #
# PyG path
# --------------------------------------------------------------------------- #
class _PyGGAT(nn.Module):
    """Two-layer GATConv stack; exposes last-layer attention per edge."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, heads: int):
        super().__init__()
        self.g1 = GATConv(in_dim, hidden, heads=heads, concat=True)
        self.g2 = GATConv(hidden * heads, out_dim, heads=1, concat=False)

    def forward(self, x, edge_index):
        h = F.elu(self.g1(x, edge_index))
        # return_attention_weights gives (edge_index, alpha) for the readout.
        h, (att_edge_index, att_w) = self.g2(
            h, edge_index, return_attention_weights=True)
        return h, (att_edge_index, att_w)


# --------------------------------------------------------------------------- #
# Fallback path (dense, plain PyTorch)
# --------------------------------------------------------------------------- #
class _DenseGATLayer(nn.Module):
    """Single-head GAT layer over a dense adjacency (Velickovic et al., 2018)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Linear(out_dim, 1, bias=False)
        self.a_dst = nn.Linear(out_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        # x: (N, in_dim), adj: (N, N) with 1 where an edge exists (self-loops ok).
        h = self.W(x)                               # (N, out)
        e = self.a_src(h) + self.a_dst(h).T          # (N, N) additive scores
        e = F.leaky_relu(e, 0.2)
        # Mask non-edges to -inf before softmax so attention stays on-graph.
        mask = adj <= 0
        e = e.masked_fill(mask, float("-inf"))
        att = torch.softmax(e, dim=-1)               # row-normalized attention
        att = torch.nan_to_num(att)                  # isolated nodes -> 0 row
        return att @ h, att


class _DenseGAT(nn.Module):
    """Two dense GAT layers; returns embeddings + the final attention matrix."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.l1 = _DenseGATLayer(in_dim, hidden)
        self.l2 = _DenseGATLayer(hidden, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        h, _ = self.l1(x, adj)
        h = F.elu(h)
        h, att = self.l2(h, adj)
        return h, att


# --------------------------------------------------------------------------- #
# Unified module
# --------------------------------------------------------------------------- #
class TopologyGAT(nn.Module):
    """Topology-aware GAT with a readable attention/causal-chain output.

    Args:
        in_dim:  per-node feature dim.
        hidden:  hidden width.
        out_dim: output embedding dim per node.
        heads:   attention heads (PyG path only; the fallback is single-head).

    ``forward`` accepts either a PyG ``edge_index`` (2, E) or a dense adjacency
    ``(N, N)``; the right one is used for the active backend. It returns
    ``(node_embeddings, attention)`` where ``attention`` is a dense ``(N, N)``
    matrix in both paths (PyG edge attention is scattered back to dense) so the
    causal-chain readout is backend-independent.
    """

    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 64,
                 heads: int = 4):
        super().__init__()
        self.use_pyg = _HAS_PYG
        if self.use_pyg:
            self._net = _PyGGAT(in_dim, hidden, out_dim, heads)
            print("[gat] using torch_geometric GATConv (sparse) path")
        else:
            self._net = _DenseGAT(in_dim, hidden, out_dim)
            print("[gat] torch_geometric not found - using dense fallback "
                  "attention path")

    def forward(self, x: torch.Tensor, graph):
        """graph = edge_index (2, E) for PyG, or dense adjacency (N, N)."""
        if self.use_pyg:
            h, (att_edge_index, att_w) = self._net(x, graph)
            att = self._scatter_attention(att_edge_index, att_w, x.shape[0])
            return h, att
        return self._net(x, graph)

    @staticmethod
    def _scatter_attention(edge_index, att_w, n_nodes: int) -> torch.Tensor:
        """Scatter PyG per-edge attention into a dense (N, N) matrix.

        Averages heads if attention is multi-head, so the result is a single
        (N, N) weight matrix comparable to the fallback path.
        """
        if att_w.dim() > 1:
            att_w = att_w.mean(dim=-1)
        att = torch.zeros(n_nodes, n_nodes, device=att_w.device, dtype=att_w.dtype)
        src, dst = edge_index[0], edge_index[1]
        att[dst, src] = att_w     # row=target node, col=source it attended to
        return att

    @staticmethod
    def causal_chain(att: torch.Tensor, node: int, depth: int = 3) -> list[int]:
        """Trace the highest-attention path back from ``node``.

        Greedily follows, at each hop, the neighbour the current node attended
        to most strongly (excluding self), up to ``depth`` hops. The returned
        list ``[node, parent, grandparent, ...]`` is the inferred propagation
        chain - the "why" behind a predicted fault.
        """
        chain = [int(node)]
        cur = int(node)
        seen = {cur}
        for _ in range(depth):
            row = att[cur].clone()
            row[cur] = float("-inf")          # don't walk the self-loop
            for s in seen:
                row[s] = float("-inf")        # avoid cycles
            if torch.isneginf(row).all() or float(row.max()) <= 0:
                break
            nxt = int(row.argmax())
            chain.append(nxt)
            seen.add(nxt)
            cur = nxt
        return chain
