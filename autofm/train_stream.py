"""Streaming training entry point (lazy windowing, low host RAM).

Same pipeline semantics as :mod:`autofm.pipeline` - load -> label -> features
-> compress (MAE/PCA) -> classify -> [conformal] - but windows are never
materialized into one ``(N, k, F)`` array. Instead
:func:`autofm.data.features.prepare_streaming` yields lazy ``WindowDataset``
splits, and both the MAE and the temporal classifier are trained by **iterating
DataLoaders**, so peak host memory is O(dataset rows), not O(windows x k x F).

Single-process only for now (no torchrun needed); GAT is off. Runs on CPU / MPS
/ CUDA, eager mode.

Run:
    python -m autofm.train_stream --config config_local.yaml --use-conformal \
        --compressor mae --backbone patchtst --head evidential \
        --max-days 6 --max-gnbs 12
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from autofm.data.features import prepare_streaming
from autofm.data.labels import CLASS_NAMES, label_frame, load_alarms
from autofm.data.loader import load
from autofm.data.window_dataset import collate_windows
from autofm.metrics import compute_all, format_summary
from autofm.model.classifier import _TemporalEncoder
from autofm.model.compress import _MAENet
from autofm.model.conformal import ConformalPredictor
from autofm.model.evidential import edl_mse_loss, evidence_to_alpha, dirichlet_probs
from autofm.pipeline import load_config


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _loader(ds, cfg, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds, batch_size=cfg.get("batch_size", 512), shuffle=shuffle,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=(torch.cuda.is_available()),
        persistent_workers=(cfg.get("num_workers", 0) > 0),
        collate_fn=collate_windows,
    )


# --------------------------------------------------------------------------- #
# Stage 1: MAE trained by streaming per-ROP rows out of the window loader
# --------------------------------------------------------------------------- #
def _train_mae_stream(cfg, train_loader, in_dim, device, bench_out, log) -> _MAENet:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    is_cuda = device.type == "cuda"
    net = _MAENet(in_dim, cfg["latent_dim"]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.get("mae_lr", 1e-3))
    loss_fn = nn.MSELoss()
    use_amp = bool(cfg.get("amp", True)) and is_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    mask_frac = cfg.get("mask_frac", 0.4)

    bench = bench_out is not None
    bench_warmup = int(cfg.get("bench_warmup", 10))
    step_ms: list[float] = []
    gstep = 0

    net.train()
    for epoch in range(cfg["mae_epochs"]):
        total, seen = 0.0, 0
        for xb, _yb, _yc, _cells in train_loader:
            # (B, k, F) -> (B*k, F): the MAE sees per-ROP rows, streamed. Only
            # one batch of windows is ever in memory at a time.
            flat = xb.reshape(-1, xb.shape[-1]).to(device, non_blocking=is_cuda)

            timed = bench and gstep >= bench_warmup
            if timed and is_cuda:
                torch.cuda.synchronize(device)
            t0 = time.perf_counter() if timed else 0.0

            mask = (torch.rand_like(flat) > mask_frac).float()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=use_amp):
                recon = net(flat * mask)
                loss = loss_fn(recon, flat)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(flat)
            seen += len(flat)

            if timed:
                if is_cuda:
                    torch.cuda.synchronize(device)
                step_ms.append((time.perf_counter() - t0) * 1000.0)
            gstep += 1
        log(f"  [MAE] epoch {epoch + 1}/{cfg['mae_epochs']} "
            f"recon_mse={total / max(seen, 1):.4f}")

    if bench and step_ms:
        median_ms = statistics.median(step_ms)
        rate = cfg.get("batch_size", 512) / (median_ms / 1000.0)
        payload = {"world_size": 1, "gpus": 1,
                   "median_step_ms": round(median_ms, 3),
                   "samples_per_sec": round(rate, 1)}
        Path(bench_out).parent.mkdir(parents=True, exist_ok=True)
        Path(bench_out).write_text(json.dumps(payload, indent=2))
        log(f"[bench] median {median_ms:.2f} ms/step  {rate:.0f} samples/s "
            f"-> {bench_out}")
    net.eval()
    return net


@torch.no_grad()
def _encode_window(mae: _MAENet, xb: torch.Tensor) -> torch.Tensor:
    """Compress a window batch (B, k, F) -> (B, k, latent) via the MAE encoder."""
    b, k, f = xb.shape
    z = mae.encoder(xb.reshape(b * k, f))
    return z.reshape(b, k, -1)


# --------------------------------------------------------------------------- #
# Stage 2: classifier trained by streaming windows -> MAE-encode -> loss
# --------------------------------------------------------------------------- #
def _class_weights(labels: np.ndarray, n_classes: int, device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32, device=device)


def _train_clf_stream(cfg, train_loader, mae, train_labels, device, log) -> _TemporalEncoder:
    head = cfg.get("clf_head", "softmax")
    n_classes = len(CLASS_NAMES)
    net = _TemporalEncoder(
        cfg["latent_dim"], n_classes, head=head,
        backbone=cfg.get("clf_backbone", "transformer"),
        window_k=cfg["window_k"], patch_size=cfg.get("patch_size", 5),
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    cw = _class_weights(train_labels, n_classes, device)
    ce = nn.CrossEntropyLoss(weight=cw)
    is_cuda = device.type == "cuda"
    anneal_epochs = int(cfg.get("anneal_epochs", 10))

    net.train()
    for epoch in range(1, cfg["clf_epochs"] + 1):
        anneal = min(1.0, epoch / max(anneal_epochs, 1))
        total, seen = 0.0, 0
        for xb, _yb, yc, _cells in train_loader:
            xb = xb.to(device, non_blocking=is_cuda)
            yc = yc.to(device, non_blocking=is_cuda)
            z = _encode_window(mae, xb)          # compress on the fly (no grad)
            out = net(z)
            if head == "evidential":
                loss = edl_mse_loss(out, yc, n_classes=n_classes,
                                    anneal=anneal, class_weight=cw)
            else:
                loss = ce(out, yc)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
            seen += len(xb)
        log(f"  [clf:{head}] epoch {epoch:02d}/{cfg['clf_epochs']} "
            f"loss={total / max(seen, 1):.4f}")
    net.eval()
    return net


# --------------------------------------------------------------------------- #
# Eval: stream a split -> probabilities (small), derive preds + uncertainty
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _predict_proba_stream(net, mae, loader, head, n_classes, device):
    """Stream a split through MAE+classifier, collecting (N, C) probabilities."""
    is_cuda = device.type == "cuda"
    probs, ys = [], []
    for xb, _yb, yc, _cells in loader:
        xb = xb.to(device, non_blocking=is_cuda)
        z = _encode_window(mae, xb)
        out = net(z)
        if head == "evidential":
            alpha = evidence_to_alpha(out)
            p = dirichlet_probs(alpha)
        else:
            p = torch.softmax(out, dim=1)
        probs.append(p.cpu().numpy())
        ys.append(yc.numpy())
    if not probs:
        return (np.empty((0, n_classes), dtype=np.float32),
                np.empty(0, dtype=np.int64))
    return np.concatenate(probs), np.concatenate(ys)


def run_stream(cfg: dict, bench_out: str | None = None) -> dict:
    t0 = time.perf_counter()
    device = _device()

    def log(*a):
        print(*a, flush=True)

    torch.manual_seed(cfg["seed"])
    root = cfg["dataset_root"]

    log(f"[1/5] Loading telemetry (max_days={cfg['max_days']}, "
        f"max_gnbs={cfg['max_gnbs']}) ...")
    d = load(root, max_days=cfg["max_days"], max_gnbs=cfg["max_gnbs"])

    log(f"[2/5] Labeling from alarms (horizon={cfg['horizon']} ROPs) ...")
    alarms = load_alarms(str(Path(root) / "alarms" / "alarms.csv"))
    labeled = label_frame(d.frame, alarms, horizon=cfg["horizon"])

    log(f"[3/5] Streaming windows (k={cfg['window_k']}) + leakage-safe split ...")
    prep = prepare_streaming(labeled, d.feature_cols, k=cfg["window_k"])
    log(f"      train {len(prep.train)} / val {len(prep.val)} / "
        f"test {len(prep.test)} windows  ({len(prep.feature_cols)} features, "
        f"streamed - not materialized)")
    if len(prep.train) == 0:
        raise SystemExit("No training windows; raise max_days/max_gnbs or lower window_k.")

    train_loader = _loader(prep.train, cfg, shuffle=True)
    val_loader = _loader(prep.val, cfg, shuffle=False)
    test_loader = _loader(prep.test, cfg, shuffle=False)
    in_dim = prep.train.n_features

    if cfg.get("compressor", "mae") != "mae":
        # PCA would require a materialized fit; the streaming path targets MAE.
        log("[warn] train_stream supports the MAE compressor; forcing mae.")
        cfg["compressor"] = "mae"

    log(f"[4/5] Training MAE ({cfg['mae_epochs']} epochs, streamed) ...")
    mae = _train_mae_stream(cfg, train_loader, in_dim, device, bench_out, log)

    head = cfg.get("clf_head", "softmax")
    backbone = cfg.get("clf_backbone", "transformer")
    log(f"[5/5] Training classifier ({cfg['clf_epochs']} epochs, "
        f"backbone={backbone}, head={head}, streamed) ...")
    train_labels = prep.train.all_class_labels()
    clf = _train_clf_stream(cfg, train_loader, mae, train_labels, device, log)

    # ---- Eval (stream test + val; only small probability arrays kept) ------
    n_classes = len(CLASS_NAMES)
    proba, y_test = _predict_proba_stream(clf, mae, test_loader, head, n_classes, device)
    pred = proba.argmax(axis=1) if len(proba) else np.empty(0, dtype=np.int64)
    # Uncertainty: evidential vacuity, else 1 - max prob (confidence gap).
    if head == "evidential" and len(proba):
        # recompute vacuity from probs is not exact; approximate with 1-maxprob
        unc = 1.0 - proba.max(axis=1)
    else:
        unc = (1.0 - proba.max(axis=1)) if len(proba) else np.empty(0)
    mean_unc = float(unc.mean()) if len(unc) else float("nan")

    metrics = compute_all(y_test, pred, CLASS_NAMES, proba=proba, uncertainty=unc)
    test_f1 = metrics["multiclass"]["macro_f1"]

    conformal_coverage = conformal_set_size = None
    if cfg.get("use_conformal", False):
        proba_val, y_val = _predict_proba_stream(
            clf, mae, val_loader, head, n_classes, device)
        if len(proba_val) and len(proba):
            cp = ConformalPredictor(confidence_level=0.95)
            cp.calibrate(proba_val, y_val)
            sets = cp.predict_set(proba)
            conformal_coverage = cp.empirical_coverage(sets, y_test)
            conformal_set_size = cp.mean_set_size(sets)
            metrics["conformal"] = {
                "target_coverage": 0.95,
                "empirical_coverage": conformal_coverage,
                "mean_set_size": conformal_set_size,
            }

    present = sorted(set(y_test.tolist()) | set(pred.tolist()))
    report = classification_report(
        y_test, pred, labels=present,
        target_names=[CLASS_NAMES[i] for i in present], zero_division=0)
    elapsed = time.perf_counter() - t0

    log("\n" + "=" * 64)
    log(f"autofm FM pipeline (streaming) - TEST macro-F1 = {test_f1:.4f}  "
        f"(compressor=mae, backbone={backbone}, head={head}, "
        f"horizon={cfg['horizon']})")
    log(f"mean uncertainty: {mean_unc:.4f}")
    log("=" * 64)
    log(format_summary(metrics, CLASS_NAMES))
    log("\nPer-class report:")
    log(report)
    if conformal_coverage is not None:
        log(f"[conformal] target 95% coverage | "
            f"empirical {conformal_coverage * 100:.1f}% | "
            f"mean set size {conformal_set_size:.2f}")
    log(f"Total wall time: {elapsed:.1f}s on device {device}")

    metrics_out = cfg.get("metrics_out")
    if metrics_out:
        payload = dict(metrics)
        payload["conformal_coverage"] = conformal_coverage
        payload["conformal_set_size"] = conformal_set_size
        payload["streaming"] = True
        payload["config"] = {k: cfg[k] for k in (
            "compressor", "clf_backbone", "clf_head", "horizon", "window_k",
            "latent_dim", "max_days", "max_gnbs") if k in cfg}
        payload["elapsed_sec"] = round(elapsed, 1)
        Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
        Path(metrics_out).write_text(json.dumps(payload, indent=2))
        log(f"[metrics] wrote {metrics_out}")

    return {"test_macro_f1": test_f1, "mean_uncertainty": mean_unc,
            "conformal_coverage": conformal_coverage,
            "conformal_set_size": conformal_set_size, "elapsed_sec": elapsed}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run the autofm pipeline with streaming (lazy) windowing.")
    ap.add_argument("--config", default="config_autofm.yaml")
    ap.add_argument("--compressor", choices=["pca", "mae"], default=None)
    ap.add_argument("--backbone", choices=["transformer", "patchtst"], default=None)
    ap.add_argument("--head", choices=["softmax", "evidential"], default=None)
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--max-gnbs", type=int, default=None)
    ap.add_argument("--use-conformal", action="store_true", default=False)
    ap.add_argument("--metrics-out", default=None)
    ap.add_argument("--bench-out", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.compressor:
        cfg["compressor"] = args.compressor
    if args.backbone:
        cfg["clf_backbone"] = args.backbone
    if args.head:
        cfg["clf_head"] = args.head
    if args.max_days is not None:
        cfg["max_days"] = args.max_days
    if args.max_gnbs is not None:
        cfg["max_gnbs"] = args.max_gnbs
    if args.use_conformal:
        cfg["use_conformal"] = True
    if args.metrics_out is not None:
        cfg["metrics_out"] = args.metrics_out

    run_stream(cfg, bench_out=args.bench_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
