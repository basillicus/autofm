"""Training-speed benchmark + scaling chart.

Measures MAE training throughput and writes a JSON result, so single-GPU and
multi-GPU runs can be compared and a scaling chart drawn. Runs on CPU/MPS too
(for local sanity), but the headline numbers are meant for CUDA.

Two sub-commands:

    # time N steps, write a JSON
    python -m autofm.bench run --gpus 1 --out artifacts/bench_1gpu.json
    python -m autofm.bench run --gpus 1 --compare   # AMP off vs on

    # draw the scaling chart from >=2 result JSONs
    python -m autofm.bench plot artifacts/bench_1gpu.json artifacts/bench_4gpu.json \
        --out artifacts/scaling.png

Timing: a few warmup steps (not counted), then ``--steps`` timed steps with
``torch.cuda.synchronize()`` bracketing each step on CUDA (so async kernels are
actually finished before we stop the clock). We report the **median** step time
(robust to stragglers) and samples/second.

Scaling efficiency for an N-GPU run vs the 1-GPU baseline:

    efficiency % = throughput_N / (N * throughput_1) * 100

i.e. 100% is perfect linear scaling.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from autofm.data.features import prepare
from autofm.data.labels import label_frame, load_alarms
from autofm.data.loader import load
from autofm.model.compress import _MAENet, _flatten_time
from autofm.pipeline import load_config


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_batches(cfg: dict, device: torch.device, batch_size: int) -> DataLoader:
    """Build a DataLoader of real windowed feature rows for timing."""
    d = load(cfg["dataset_root"], max_days=cfg["max_days"], max_gnbs=cfg["max_gnbs"])
    labeled = label_frame(
        d.frame,
        load_alarms(str(Path(cfg["dataset_root"]) / "alarms" / "alarms.csv")),
        horizon=cfg["horizon"],
    )
    prep = prepare(labeled, d.feature_cols, k=cfg["window_k"])
    flat = _flatten_time(prep.train.X).astype(np.float32)
    g = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(flat)),
        batch_size=batch_size, shuffle=True, generator=g,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.get("num_workers", 0) > 0),
        drop_last=True,
    )
    return loader, flat.shape[1]


def _time_training(
    cfg: dict, *, amp: bool, warmup: int, steps: int,
    batch_size: int, latent_dim: int,
) -> dict:
    """Run warmup + timed steps, return timing stats for one configuration."""
    device = _device()
    is_cuda = device.type == "cuda"
    use_amp = bool(amp) and is_cuda

    loader, in_dim = _load_batches(cfg, device, batch_size)
    net = _MAENet(in_dim, latent_dim).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    mask_frac = cfg.get("mask_frac", 0.4)

    def one_step(xb: torch.Tensor) -> None:
        xb = xb.to(device, non_blocking=is_cuda)
        mask = (torch.rand_like(xb) > mask_frac).float()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            loss = loss_fn(net(xb * mask), xb)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    net.train()
    it = iter(loader)

    def next_batch():
        nonlocal it
        try:
            return next(it)[0]
        except StopIteration:
            it = iter(loader)
            return next(it)[0]

    # Warmup (not timed): lets cuDNN autotune / allocator settle.
    for _ in range(warmup):
        one_step(next_batch())
    if is_cuda:
        torch.cuda.synchronize(device)

    # Timed steps.
    step_ms: list[float] = []
    for _ in range(steps):
        xb = next_batch()
        if is_cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        one_step(xb)
        if is_cuda:
            torch.cuda.synchronize(device)
        step_ms.append((time.perf_counter() - t0) * 1000.0)

    median_ms = statistics.median(step_ms)
    samples_per_s = batch_size / (median_ms / 1000.0)
    return {
        "device": str(device),
        "device_name": (torch.cuda.get_device_name(device) if is_cuda else str(device)),
        "amp": use_amp,
        "batch_size": batch_size,
        "warmup": warmup,
        "steps": steps,
        "median_step_ms": round(median_ms, 3),
        "samples_per_sec": round(samples_per_s, 1),
    }


def cmd_run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.max_days is not None:
        cfg["max_days"] = args.max_days
    if args.max_gnbs is not None:
        cfg["max_gnbs"] = args.max_gnbs

    # Precedence: explicit --batch-size flag > config batch_size > default.
    if args.batch_size is not None:
        batch_size, bs_source = args.batch_size, "flag"
    elif "batch_size" in cfg:
        batch_size, bs_source = cfg["batch_size"], "config"
    else:
        batch_size, bs_source = 512, "default"
    print(f"[bench] batch_size={batch_size} (source: {bs_source})")

    configs = [False, True] if args.compare else [not args.no_amp]
    runs = []
    for amp in configs:
        res = _time_training(
            cfg, amp=amp, warmup=args.warmup, steps=args.steps,
            batch_size=batch_size, latent_dim=cfg["latent_dim"],
        )
        tag = "AMP" if res["amp"] else "no-AMP"
        print(f"[bench] {tag:6s} median {res['median_step_ms']:.2f} ms/step  "
              f"{res['samples_per_sec']:.0f} samples/s  on {res['device_name']}")
        runs.append(res)

    out = {
        "gpus": args.gpus,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "config": {k: cfg[k] for k in ("max_days", "max_gnbs", "window_k",
                                        "latent_dim")},
        "runs": runs,
        # Headline throughput = the fastest run (AMP-on when compared).
        "samples_per_sec": max(r["samples_per_sec"] for r in runs),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[bench] wrote {out_path}")

    if args.compare and len(runs) == 2:
        off, on = runs[0]["samples_per_sec"], runs[1]["samples_per_sec"]
        if off > 0:
            print(f"[bench] AMP speedup: {on / off:.2f}x "
                  f"({off:.0f} -> {on:.0f} samples/s)")


def cmd_plot(args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")  # headless (no display on a cluster / CI)
    import matplotlib.pyplot as plt

    results = []
    for p in args.results:
        data = json.loads(Path(p).read_text())
        results.append((int(data["gpus"]), float(data["samples_per_sec"])))
    results.sort(key=lambda t: t[0])

    gpus = [g for g, _ in results]
    thru = [t for _, t in results]
    base_gpus, base_thru = results[0]  # smallest gpu count is the baseline

    print("GPUs | samples/s | scaling efficiency %")
    eff: list[float] = []
    for g, t in results:
        e = t / (g / base_gpus * base_thru) * 100.0 if base_thru else 0.0
        eff.append(e)
        print(f"{g:4d} | {t:9.0f} | {e:6.1f}")

    # Chart: measured throughput vs ideal linear scaling.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(gpus, thru, "o-", label="measured", linewidth=2)
    ideal = [base_thru * (g / base_gpus) for g in gpus]
    ax.plot(gpus, ideal, "--", color="gray", label="ideal linear")
    for g, t, e in zip(gpus, thru, eff):
        ax.annotate(f"{e:.0f}%", (g, t), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("number of GPUs")
    ax.set_ylabel("throughput (samples/sec)")
    ax.set_title("autofm MAE - DDP scaling")
    ax.set_xticks(gpus)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"[bench] wrote {out_path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="autofm training benchmark + scaling.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Time training steps and write a JSON.")
    r.add_argument("--config", default="config_autofm.yaml")
    r.add_argument("--gpus", type=int, default=1,
                   help="GPU count this run represents (for the scaling chart).")
    r.add_argument("--out", default="artifacts/bench.json")
    r.add_argument("--steps", type=int, default=50)
    r.add_argument("--warmup", type=int, default=10)
    r.add_argument("--batch-size", type=int, default=None,
                   help="Batch size; falls back to config batch_size, then 512.")
    r.add_argument("--compare", action="store_true",
                   help="Run AMP off and on, report the speedup.")
    r.add_argument("--no-amp", action="store_true",
                   help="Disable AMP for a single (non-compare) run.")
    r.add_argument("--max-days", type=int, default=None)
    r.add_argument("--max-gnbs", type=int, default=None)
    r.set_defaults(func=cmd_run)

    p = sub.add_parser("plot", help="Draw the scaling chart from result JSONs.")
    p.add_argument("results", nargs="+", help="Two+ bench JSON files.")
    p.add_argument("--out", default="artifacts/scaling.png")
    p.set_defaults(func=cmd_plot)

    args = ap.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
