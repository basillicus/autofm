"""Feature matrix, windowing, and leakage-safe splits.

Turns the labeled cell-centric frame (loader + labels) into model-ready tensors:

1. **Clean features** - drop all-null columns, forward/back-fill sparse pm
   counters per cell, fill any remainder with 0.
2. **Chronological split** - 60/20/20 by ROP time; val and test strictly after
   train (no shuffling across the boundary).
3. **Standardize** - per-feature mean/std fit on the **train split only**, then
   applied to val/test (the design-doc leakage rule).
4. **Window** - per cell, sliding ``k``-ROP lookback windows -> ``(N, k, F)``.
   The label is the window's final-step ``will_fault`` / ``fault_class`` (which
   already encodes "within the next ``horizon`` ROPs", set at label time).

The windower never crosses a cell boundary, so a window is always one cell's
contiguous history.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TS = "rdiTimestampTZ"


@dataclass
class Dataset:
    """Windowed tensors for one split."""

    X: np.ndarray            # (N, k, F) float32
    y_binary: np.ndarray     # (N,) int8   - will_fault
    y_class: np.ndarray      # (N,) int64  - fault_class
    cells: np.ndarray        # (N,) object - originating cell id (for grouping)


@dataclass
class Prepared:
    """All splits + the fitted scaling stats and feature names."""

    train: Dataset
    val: Dataset
    test: Dataset
    feature_cols: list[str]
    mean: np.ndarray
    std: np.ndarray


def _clean_features(frame: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Drop all-null columns; fill sparse counters per cell, then zero-fill."""
    f = frame.copy()
    # Drop columns that are entirely null across the loaded data.
    keep = [c for c in feature_cols if f[c].notna().any()]
    # Per-cell forward then backward fill (dense 15-min telemetry), then 0.
    f[keep] = (f.groupby("cell")[keep]
               .transform(lambda s: s.ffill().bfill()))
    f[keep] = f[keep].fillna(0.0)
    return f, keep


def _time_split_bounds(timestamps: np.ndarray,
                       train_frac: float, val_frac: float) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the (train_end, val_end) timestamps for a chronological split.

    Indices are clamped to ``[0, n-1]`` so that very small numbers of distinct
    ROPs (e.g. smoke-scale runs) never underflow to ``-1`` and silently wrap to
    the last timestamp, which would corrupt the split boundaries.
    """
    ts_sorted = np.sort(np.unique(timestamps))
    n = len(ts_sorted)
    if n == 0:
        raise ValueError("No timestamps to split on")
    train_idx = min(max(int(n * train_frac) - 1, 0), n - 1)
    val_idx = min(max(int(n * (train_frac + val_frac)) - 1, 0), n - 1)
    train_end = ts_sorted[train_idx]
    val_end = ts_sorted[val_idx]
    return train_end, val_end


def _standardize_fit(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    std[std == 0] = 1.0  # guard constant columns
    return mean.astype(np.float32), std.astype(np.float32)


def _make_windows(
    f: pd.DataFrame, feature_cols: list[str], k: int,
    mean: np.ndarray, std: np.ndarray,
) -> Dataset:
    """Build (N, k, F) windows per cell from a time-sorted frame slice."""
    Xs, yb, yc, cs = [], [], [], []
    feat_idx = feature_cols
    for cell, g in f.groupby("cell", sort=False):
        g = g.sort_values(TS)
        vals = ((g[feat_idx].to_numpy(dtype=np.float32) - mean) / std)
        wf = g["will_fault"].to_numpy(dtype=np.int8)
        fc = g["fault_class"].to_numpy(dtype=np.int64)
        n = len(g)
        if n < k:
            continue
        for end in range(k, n + 1):          # window = [end-k, end)
            Xs.append(vals[end - k:end])
            yb.append(wf[end - 1])           # label at the window's last step
            yc.append(fc[end - 1])
            cs.append(cell)
    if not Xs:
        empty = np.empty((0, k, len(feat_idx)), dtype=np.float32)
        return Dataset(empty, np.empty(0, np.int8), np.empty(0, np.int64),
                       np.empty(0, object))
    return Dataset(
        X=np.stack(Xs).astype(np.float32),
        y_binary=np.asarray(yb, dtype=np.int8),
        y_class=np.asarray(yc, dtype=np.int64),
        cells=np.asarray(cs, dtype=object),
    )


@dataclass
class PreparedStreaming:
    """Streaming splits: lazy WindowDatasets + the fitted scaling stats."""

    train: "object"          # WindowDataset (imported lazily to avoid torch dep)
    val: "object"
    test: "object"
    feature_cols: list[str]
    mean: np.ndarray
    std: np.ndarray


def _per_cell_arrays(f: pd.DataFrame, feature_cols: list[str],
                     mean: np.ndarray, std: np.ndarray) -> tuple[dict, dict, dict]:
    """Build per-cell standardized (T,F) arrays + label arrays for a split.

    This is the only data held in memory: O(rows x F), NOT O(windows x k x F).
    Windows are sliced from these arrays lazily by :class:`WindowDataset`.
    """
    vals: dict = {}
    will_fault: dict = {}
    fault_class: dict = {}
    for cell, g in f.groupby("cell", sort=False):
        g = g.sort_values(TS)
        vals[cell] = ((g[feature_cols].to_numpy(dtype=np.float32) - mean) / std
                      ).astype(np.float32)
        will_fault[cell] = g["will_fault"].to_numpy(dtype=np.int8)
        fault_class[cell] = g["fault_class"].to_numpy(dtype=np.int64)
    return vals, will_fault, fault_class


def prepare_streaming(
    labeled_frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    k: int = 60,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> "PreparedStreaming":
    """Like :func:`prepare`, but returns lazy ``WindowDataset`` splits.

    Reuses the cheap clean / chronological-split / train-only-standardize logic
    (identical semantics to ``prepare``), then - instead of materializing every
    window via ``_make_windows`` - stores per-cell standardized arrays and a
    tiny window index inside three ``WindowDataset`` objects. Peak memory is
    O(dataset rows), fixing the eager-path OOM.
    """
    from autofm.data.window_dataset import WindowDataset

    f, keep = _clean_features(labeled_frame, feature_cols)
    f = f.sort_values(["cell", TS]).reset_index(drop=True)

    train_end, val_end = _time_split_bounds(
        f[TS].to_numpy(), train_frac, val_frac)

    train_f = f[f[TS] <= train_end]
    val_f = f[(f[TS] > train_end) & (f[TS] <= val_end)]
    test_f = f[f[TS] > val_end]

    # Fit scaler on train rows only (leakage-safe) - same as prepare().
    mean, std = _standardize_fit(train_f[keep].to_numpy(dtype=np.float32))

    def _ds(split_f: pd.DataFrame) -> WindowDataset:
        vals, wf, fc = _per_cell_arrays(split_f, keep, mean, std)
        return WindowDataset(vals, wf, fc, k)

    return PreparedStreaming(
        train=_ds(train_f), val=_ds(val_f), test=_ds(test_f),
        feature_cols=keep, mean=mean, std=std,
    )


def prepare(
    labeled_frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    k: int = 60,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> Prepared:
    """Clean, split (by time), scale (train-fit), and window the labeled frame.

    Args:
        labeled_frame: output of ``label_frame`` (has will_fault, fault_class).
        feature_cols:  numeric feature columns to use as model inputs.
        k:             lookback window length in ROPs.
        train_frac/val_frac: chronological split fractions (test is the rest).
    """
    f, keep = _clean_features(labeled_frame, feature_cols)
    f = f.sort_values(["cell", TS]).reset_index(drop=True)

    train_end, val_end = _time_split_bounds(
        f[TS].to_numpy(), train_frac, val_frac)

    train_f = f[f[TS] <= train_end]
    val_f = f[(f[TS] > train_end) & (f[TS] <= val_end)]
    test_f = f[f[TS] > val_end]

    # Fit scaler on train rows only (leakage-safe).
    mean, std = _standardize_fit(train_f[keep].to_numpy(dtype=np.float32))

    train = _make_windows(train_f, keep, k, mean, std)
    val = _make_windows(val_f, keep, k, mean, std)
    test = _make_windows(test_f, keep, k, mean, std)

    return Prepared(train=train, val=val, test=test, feature_cols=keep,
                    mean=mean, std=std)
