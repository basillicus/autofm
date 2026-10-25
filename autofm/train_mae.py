"""DDP-capable MAE training entry point.

Trains the masked autoencoder (the autofm compression stage) with PyTorch
DistributedDataParallel so it scales across the 4x A100 of a Leonardo node,
while still running as a plain single process on a laptop (CPU/MPS) when
launched without ``torchrun``.

Launch:
    # single process (laptop / single GPU)
    python -m autofm.train_mae --config config_autofm.yaml

    # multi-GPU on one node
    torchrun --nproc_per_node=4 -m autofm.train_mae --config config_autofm.yaml

Distribution is driven by the standard ``torchrun`` environment variables
(``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE``). When ``WORLD_SIZE`` is unset or 1
we skip the process group entirely, so there is no NCCL/CUDA dependency on a
Mac. Only rank 0 prints, evaluates, and writes the checkpoint; the AMP/NVTX
approach mirrors :meth:`autofm.model.compress.MAECompressor.fit` (CUDA-only),
and each process keeps its own ``GradScaler``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from autofm.data.features import prepare
from autofm.data.labels import label_frame, load_alarms
from autofm.data.loader import load
from autofm.model.compress import _MAENet, _flatten_time
from autofm.pipeline import load_config


# --------------------------------------------------------------------------- #
# Distributed helpers
# --------------------------------------------------------------------------- #
def _dist_info() -> tuple[int, int, int, bool]:
    """Read (rank, local_rank, world_size, is_distributed) from the env.

    ``torchrun`` sets RANK/LOCAL_RANK/WORLD_SIZE. Absent (or WORLD_SIZE=1) we
    run as a single plain process - the laptop path.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, local_rank, world_size, world_size > 1


def _pick_device(local_rank: int) -> torch.device:
    """cuda:LOCAL_RANK when CUDA is present, else MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _setup(is_distributed: bool, local_rank: int) -> torch.device:
    """Init the process group (NCCL) for DDP; no-op in single-process mode."""
    if is_distributed:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return _pick_device(local_rank)


def _cleanup(is_distributed: bool) -> None:
    if is_distributed:
        torch.distributed.destroy_process_group()


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(cfg: dict) -> None:
    rank, local_rank, world_size, is_distributed = _dist_info()
    device = _setup(is_distributed, local_rank)
    is_main = rank == 0
    is_cuda = device.type == "cuda"

    def log(*a):
        if is_main:
            print(*a, flush=True)

    # TF32 (Ampere+); harmless elsewhere.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(cfg["seed"])

    # ---- Data (every rank loads the same smoke/full slice) -----------------
    log(f"[data] loading telemetry (max_days={cfg['max_days']}, "
        f"max_gnbs={cfg['max_gnbs']}) ...")
    d = load(cfg["dataset_root"], max_days=cfg["max_days"], max_gnbs=cfg["max_gnbs"])
    labeled = label_frame(
        d.frame,
        load_alarms(str(Path(cfg["dataset_root"]) / "alarms" / "alarms.csv")),
        horizon=cfg["horizon"],
    )
    prep = prepare(labeled, d.feature_cols, k=cfg["window_k"])
    flat = _flatten_time(prep.train.X).astype(np.float32)
    log(f"[data] train rows {len(flat)}, feature dim {flat.shape[1]}")

    # ---- Model + DDP -------------------------------------------------------
    net = _MAENet(flat.shape[1], cfg["latent_dim"]).to(device)
    if is_distributed:
        net = nn.parallel.DistributedDataParallel(
            net, device_ids=[local_rank], output_device=local_rank)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.get("mae_lr", 1e-3))
    loss_fn = nn.MSELoss()

    use_amp = bool(cfg.get("amp", True)) and is_cuda
    pin = is_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def _push(t):
        if is_cuda:
            torch.cuda.nvtx.range_push(t)

    def _pop():
        if is_cuda:
            torch.cuda.nvtx.range_pop()

    # ---- DataLoader + DistributedSampler -----------------------------------
    dataset = TensorDataset(torch.from_numpy(flat))
    sampler = (DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                  shuffle=True, seed=cfg["seed"])
               if is_distributed else None)
    g = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(
        dataset, batch_size=cfg.get("batch_size", 512),
        shuffle=(sampler is None), sampler=sampler, generator=g,
        num_workers=cfg.get("num_workers", 0), pin_memory=pin,
        persistent_workers=(cfg.get("num_workers", 0) > 0),
    )

    mask_frac = cfg.get("mask_frac", 0.4)
    epochs = cfg["mae_epochs"]
    n = len(flat)
    net.train()
    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)       # reshuffle deterministically per epoch
        _push(f"mae/epoch_{epoch}")
        total = 0.0
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=pin)
            mask = (torch.rand_like(xb) > mask_frac).float()
            opt.zero_grad(set_to_none=True)
            _push("mae/forward")
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=use_amp):
                recon = net(xb * mask)
                loss = loss_fn(recon, xb)
            _pop()
            _push("mae/backward")
            scaler.scale(loss).backward()
            _pop()
            _push("mae/optim")
            scaler.step(opt)
            scaler.update()
            _pop()
            total += loss.item() * len(xb)
        _pop()
        # Each rank sees ~n/world_size rows; this per-rank mean is a fine
        # progress signal (rank 0 only).
        denom = n / max(world_size, 1)
        log(f"  [MAE] epoch {epoch + 1}/{epochs} recon_mse={total / denom:.4f}")

    # ---- Checkpoint + eval (rank 0 only) -----------------------------------
    if is_main:
        core = net.module if is_distributed else net
        out = Path(cfg.get("mae_ckpt", "artifacts/mae.pt"))
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": core.state_dict(),
             "in_dim": flat.shape[1],
             "latent_dim": cfg["latent_dim"]},
            out,
        )
        log(f"[ckpt] saved MAE -> {out}")

        # Quick reconstruction MSE on the val split as a sanity eval.
        core.eval()
        with torch.no_grad():
            vflat = torch.from_numpy(
                _flatten_time(prep.val.X).astype(np.float32)).to(device)
            vrecon = core(vflat)
            vmse = float(nn.functional.mse_loss(vrecon, vflat).item())
        log(f"[eval] val recon_mse={vmse:.4f}")

    _cleanup(is_distributed)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train the autofm MAE (DDP-capable).")
    ap.add_argument("--config", default="config_autofm.yaml")
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--max-gnbs", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override mae_epochs (handy for short profiling runs).")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.max_days is not None:
        cfg["max_days"] = args.max_days
    if args.max_gnbs is not None:
        cfg["max_gnbs"] = args.max_gnbs
    if args.epochs is not None:
        cfg["mae_epochs"] = args.epochs

    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
