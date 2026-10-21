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
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import classification_report

from autofm.data.features import prepare
from autofm.data.labels import CLASS_NAMES, label_frame, load_alarms
from autofm.data.loader import load
from autofm.model.classifier import TemporalClassifier
from autofm.model.compress import build_compressor

DEFAULT_CONFIG = {
    "dataset_root": "./Documents/hackathon/autofm-dataset-ie",
    "max_days": 14,
    "max_gnbs": 40,
    "horizon": 5,
    "window_k": 32,
    "latent_dim": 32,
    "compressor": "mae",          # "pca" | "mae"
    "mae_epochs": 6,
    "clf_epochs": 12,
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
    comp = build_compressor(cfg["compressor"], latent_dim=cfg["latent_dim"], **comp_kw)
    comp.fit(prep.train.X)
    Ztr = comp.transform(prep.train.X)
    Zval = comp.transform(prep.val.X)
    Zte = comp.transform(prep.test.X)
    if cfg["compressor"] == "pca":
        print(f"      PCA explained variance: {comp.explained_variance:.3f}")

    print(f"[5/5] Training temporal classifier ({cfg['clf_epochs']} epochs) ...")
    clf = TemporalClassifier(
        latent_dim=cfg["latent_dim"], n_classes=len(CLASS_NAMES),
        epochs=cfg["clf_epochs"], seed=cfg["seed"],
    )
    clf.fit(Ztr, prep.train.y_class, Zval, prep.val.y_class)

    # Final test metrics.
    test_f1 = clf.macro_f1(Zte, prep.test.y_class)
    pred = clf.predict(Zte)
    present = sorted(set(prep.test.y_class) | set(pred))
    report = classification_report(
        prep.test.y_class, pred,
        labels=present, target_names=[CLASS_NAMES[i] for i in present],
        zero_division=0,
    )
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 64)
    print(f"autofm FM pipeline - TEST macro-F1 = {test_f1:.4f}  "
          f"(compressor={cfg['compressor']}, horizon={cfg['horizon']})")
    print("=" * 64)
    print(report)
    print(f"Total wall time: {elapsed:.1f}s on device "
          f"{getattr(clf, 'device', 'cpu')}")

    return {"test_macro_f1": test_f1, "elapsed_sec": elapsed}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the autofm FM pipeline.")
    ap.add_argument("--config", default="config_autofm.yaml")
    ap.add_argument("--compressor", choices=["pca", "mae"], default=None,
                    help="Override the compressor from config.")
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--max-gnbs", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.compressor:
        cfg["compressor"] = args.compressor
    if args.max_days is not None:
        cfg["max_days"] = args.max_days
    if args.max_gnbs is not None:
        cfg["max_gnbs"] = args.max_gnbs

    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
