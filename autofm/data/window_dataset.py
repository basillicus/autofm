"""Lazy / streaming windowing - a torch Dataset that slices on demand.

The eager path (:func:`autofm.data.features._make_windows`) materializes every
sliding window into one ``(N, k, F)`` array via ``np.stack``. At full scale
that array is O(dataset) - ~123 GB at 180 days - which OOMs host RAM.

A window is just a contiguous slice of its cell's time-sorted, standardized
array, so there is no need to store them. :class:`WindowDataset` keeps:

* one standardized ``(T_cell, F)`` array per cell (the data we already have),
* a tiny integer index of ``(cell_key, end_idx)`` for every valid window.

``__getitem__`` slices ``vals[cell][end-k:end]`` on demand. Peak memory is then
O(dataset rows), not O(windows x k), and a ``DataLoader`` streams batches so
neither the MAE nor the classifier ever sees a fully materialized window array.

Per-window outputs match the eager ``Dataset`` row shapes so models are
unchanged: ``X`` is ``(k, F)`` float32, labels are the window's **last-step**
``will_fault`` / ``fault_class``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """Sliding windows over per-cell arrays, sliced lazily in ``__getitem__``.

    Args:
        vals:        dict ``cell -> (T_cell, F)`` float32 standardized array.
        will_fault:  dict ``cell -> (T_cell,)`` int8.
        fault_class: dict ``cell -> (T_cell,)`` int64.
        k:           lookback window length in ROPs.

    The index is built once: for each cell with ``T_cell >= k`` and every valid
    window end ``k..T_cell``, store ``(cell, end)``. These are plain ints, so the
    index is tiny regardless of feature dim.
    """

    def __init__(self, vals: dict, will_fault: dict, fault_class: dict, k: int):
        self.vals = vals
        self.will_fault = will_fault
        self.fault_class = fault_class
        self.k = k
        self.index: list[tuple[object, int]] = []
        for cell, arr in vals.items():
            n = len(arr)
            if n < k:
                continue
            # window = [end-k, end); label at index end-1.
            for end in range(k, n + 1):
                self.index.append((cell, end))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        cell, end = self.index[i]
        k = self.k
        x = self.vals[cell][end - k:end]                 # (k, F) view -> copy below
        x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        yb = int(self.will_fault[cell][end - 1])
        yc = int(self.fault_class[cell][end - 1])
        return x, yb, yc, cell

    # -- convenience --------------------------------------------------------
    @property
    def n_features(self) -> int:
        for arr in self.vals.values():
            return int(arr.shape[1])
        return 0

    def all_class_labels(self) -> np.ndarray:
        """Window last-step fault_class for every window (cheap; ints only).

        Useful for class-weight computation without materializing any features.
        """
        out = np.empty(len(self.index), dtype=np.int64)
        for i, (cell, end) in enumerate(self.index):
            out[i] = self.fault_class[cell][end - 1]
        return out

    def cells_array(self) -> np.ndarray:
        """Originating cell id per window (object array), order matches index."""
        return np.asarray([c for (c, _e) in self.index], dtype=object)


def collate_windows(batch):
    """Collate ``(x, yb, yc, cell)`` tuples into batched tensors + a cell list.

    Cells are kept as a Python list (strings), not stacked, since they are only
    used for grouping / advisory, never fed to the model.
    """
    xs = torch.stack([b[0] for b in batch])
    yb = torch.tensor([b[1] for b in batch], dtype=torch.int64)
    yc = torch.tensor([b[2] for b in batch], dtype=torch.int64)
    cells = [b[3] for b in batch]
    return xs, yb, yc, cells
