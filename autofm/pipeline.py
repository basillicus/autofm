"""End-to-end autofm FM pipeline (first runnable slice).

Wires the stages into one runnable path:

    load multi-level telemetry
      -> label from the alarm feed (forecast horizon)
      -> clean / split / scale / window
      -> compress each ROP to a latent (PCA or MAE)
      -> temporal classifier (PatchTST-lite) over the window
      -> report macro-F1 on the held-out test split

This is the baseline chain the richer autofm components (GAT, evidential heads,
conformal calibration) will slot into later. It runs at smoke scale on CPU/MPS
and scales up by widening the data knobs in the config.

Run:
    python -m autofm.pipeline --config config_autofm.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import classification_report

from autofm.data.features import prepare
from autofm.data.labels import CLASS_NAMES, label_frame, load_alarms
from autofm.data.loader import load
from autofm.metrics import compute_all, format_summary, write_prometheus
from autofm.model.classifier import TemporalClassifier
from autofm.model.compress import build_compressor

DEFAULT_CONFIG = {
    "dataset_root": "./data/autofm-dataset-ie",
    "max_days": 14,
    "max_gnbs": 40,
    "horizon": 5,
    "window_k": 32,
    "latent_dim": 32,
    "compressor": "mae",          # "pca" | "mae"
    "mae_epochs": 6,
    "amp": True,                  # mixed precision (CUDA only; ignored on CPU/MPS)
    "num_workers": 8,             # DataLoader workers for MAE training
    "clf_epochs": 12,
    "clf_head": "softmax",        # "softmax" (baseline) | "evidential" (+uncertainty)
    "clf_backbone": "transformer",# "transformer" | "patchtst"
    "patch_size": 5,              # patch length for the patchtst backbone
    "seed": 42,
}


def load_config(path: str | None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if path and Path(path).exists():
        with open(path) as fh:
            cfg.update(yaml.safe_load(fh) or {})
    return cfg


def run(cfg: dict) -> dict:
    t0 = time.perf_counter()
    root = cfg["dataset_root"]

    print(f"[1/5] Loading telemetry (max_days={cfg['max_days']}, "
          f"max_gnbs={cfg['max_gnbs']}) ...")
    d = load(root, max_days=cfg["max_days"], max_gnbs=cfg["max_gnbs"])
    print(f"      {len(d.frame)} rows, {len(d.feature_cols)} raw features, "
          f"{d.frame['cell'].nunique()} cells, {len(d.timestamps)} ROPs")

    print(f"[2/5] Labeling from alarms (horizon={cfg['horizon']} ROPs) ...")
    alarms = load_alarms(str(Path(root) / "alarms" / "alarms.csv"))
    labeled = label_frame(d.frame, alarms, horizon=cfg["horizon"])
    print(f"      positive rate {labeled['will_fault'].mean():.3f}")

    print(f"[3/5] Windowing (k={cfg['window_k']}) + leakage-safe split ...")
    prep = prepare(labeled, d.feature_cols, k=cfg["window_k"])
    print(f"      train {prep.train.X.shape}  val {prep.val.X.shape}  "
          f"test {prep.test.X.shape}  ({len(prep.feature_cols)} features)")

    print(f"[4/5] Compressing with '{cfg['compressor']}' "
          f"-> latent {cfg['latent_dim']} ...")
    comp_kw = {"seed": cfg["seed"]}
    if cfg["compressor"] == "mae":
        comp_kw["epochs"] = cfg["mae_epochs"]
        comp_kw["amp"] = cfg.get("amp", True)
        comp_kw["num_workers"] = cfg.get("num_workers", 0)
    comp = build_compressor(cfg["compressor"], latent_dim=cfg["latent_dim"], **comp_kw)
    comp.fit(prep.train.X)
    Ztr = comp.transform(prep.train.X)
    Zval = comp.transform(prep.val.X)
    Zte = comp.transform(prep.test.X)
    if cfg["compressor"] == "pca":
        print(f"      PCA explained variance: {comp.explained_variance:.3f}")

    head = cfg.get("clf_head", "softmax")
    backbone = cfg.get("clf_backbone", "transformer")
    print(f"[5/5] Training temporal classifier ({cfg['clf_epochs']} epochs, "
          f"backbone={backbone}, head={head}) ...")
    clf = TemporalClassifier(
        latent_dim=cfg["latent_dim"], n_classes=len(CLASS_NAMES),
        epochs=cfg["clf_epochs"], seed=cfg["seed"], head=head,
        backbone=backbone, window_k=cfg["window_k"],
        patch_size=cfg.get("patch_size", 5),
    )
    clf.fit(Ztr, prep.train.y_class, Zval, prep.val.y_class)

    # Final test metrics. One forward pass: get probabilities, derive the class
    # prediction and uncertainty from them, so pred / proba / metrics are all
    # consistent and we don't re-run the net three times.
    proba = clf.predict_proba(Zte)
    _, unc = clf.predict_with_uncertainty(Zte)
    pred = (proba.argmax(axis=1) if len(proba)
            else np.empty(0, dtype=np.int64))
    mean_unc = float(unc.mean()) if len(unc) else float("nan")

    metrics = compute_all(
        prep.test.y_class, pred, CLASS_NAMES, proba=proba, uncertainty=unc,
    )
    test_f1 = metrics["multiclass"]["macro_f1"]
    present = sorted(set(prep.test.y_class) | set(pred))
    report = classification_report(
        prep.test.y_class, pred,
        labels=present, target_names=[CLASS_NAMES[i] for i in present],
        zero_division=0,
    )
    elapsed = time.perf_counter() - t0

    unc_label = ("mean uncertainty" if head == "evidential"
                 else "mean confidence-gap")
    bf = metrics["binary_fault"]
    print("\n" + "=" * 64)
    print(f"autofm FM pipeline - TEST macro-F1 = {test_f1:.4f}  "
          f"(compressor={cfg['compressor']}, backbone={backbone}, "
          f"head={head}, horizon={cfg['horizon']})")
    print(f"{unc_label}: {mean_unc:.4f}")
    print("=" * 64)
    print(format_summary(metrics, CLASS_NAMES))
    print("\nPer-class report:")
    print(report)
    print(f"Total wall time: {elapsed:.1f}s on device "
          f"{getattr(clf, 'device', 'cpu')}")

    # Optional machine-readable metrics dump for tracking across runs.
    metrics_out = cfg.get("metrics_out")
    if metrics_out:
        payload = dict(metrics)
        payload["config"] = {k: cfg[k] for k in (
            "compressor", "clf_backbone", "clf_head", "horizon", "window_k",
            "latent_dim", "max_days", "max_gnbs") if k in cfg}
        payload["elapsed_sec"] = round(elapsed, 1)
        out_path = Path(metrics_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"[metrics] wrote {out_path}")

    # Optional Prometheus exposition file (node_exporter textfile collector).
    prom_out = cfg.get("prom_out")
    if prom_out:
        prom_labels = {
            "compressor": str(cfg.get("compressor", "")),
            "backbone": str(cfg.get("clf_backbone", "")),
            "head": str(head),
            "horizon": str(cfg.get("horizon", "")),
        }
        write_prometheus(prom_out, metrics, prom_labels)
        print(f"[metrics] wrote Prometheus exposition -> {prom_out}")

    result = {"test_macro_f1": test_f1, "mean_uncertainty": mean_unc,
              "elapsed_sec": elapsed}
    result.update({
        "weighted_f1": metrics["multiclass"]["weighted_f1"],
        "balanced_accuracy": metrics["multiclass"]["balanced_accuracy"],
        "fault_recall": bf["recall"],
        "fault_precision": bf["precision"],
        "fault_pr_auc": bf.get("pr_auc"),
        "fault_roc_auc": bf.get("roc_auc"),
        "metrics": metrics,
    })
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the autofm FM pipeline.")
    ap.add_argument("--config", default="config_autofm.yaml")
    ap.add_argument("--compressor", choices=["pca", "mae"], default=None,
                    help="Override the compressor from config.")
    ap.add_argument("--head", choices=["softmax", "evidential"], default=None,
                    help="Override the classifier head from config.")
    ap.add_argument("--backbone", choices=["transformer", "patchtst"], default=None,
                    help="Override the temporal backbone from config.")
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--max-gnbs", type=int, default=None)
    ap.add_argument("--metrics-out", default=None,
                    help="Write the full metrics bundle as JSON to this path.")
    ap.add_argument("--prom-out", default=None,
                    help="Write Prometheus exposition metrics (.prom) to this "
                         "path for the node_exporter textfile collector.")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.compressor:
        cfg["compressor"] = args.compressor
    if args.head:
        cfg["clf_head"] = args.head
    if args.backbone:
        cfg["clf_backbone"] = args.backbone
    if args.max_days is not None:
        cfg["max_days"] = args.max_days
    if args.max_gnbs is not None:
        cfg["max_gnbs"] = args.max_gnbs
    if args.metrics_out is not None:
        cfg["metrics_out"] = args.metrics_out
    if args.prom_out is not None:
        cfg["prom_out"] = args.prom_out

    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
