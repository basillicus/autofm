"""Dataset profiler.

The dataset is large (~128 GB) and initially unknown, so before building any
pipeline we survey it: directory layout, per-level file groups, column shapes,
cadence, entity counts (parsed from ``fdn``), the date range, and the alarm
feed that supplies labels. The output is a human-readable DATA_NOTES.md plus a
machine-readable profile.json the loader can consume.

Run:
    python -m autofm.profile_dataset --root /path/to/autofm-dataset-ie \
        --out docs/DATA_NOTES.md

It reads only a small sample of files (one day per level by default), so it is
cheap even against the full dataset.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Single source of truth for fdn parsing + metadata columns (see autofm/fdn.py).
from autofm.fdn import LEVELS, META_COLS, parse_fdn


@dataclass
class GroupProfile:
    """Profile of one file group (e.g. NRCellDU, OCloudNode)."""

    level: str
    group: str
    sample_file: str
    n_rows: int
    n_cols: int
    n_numeric_feature_cols: int
    feature_cols_preview: list[str]
    distinct_timestamps: int
    distinct_entities: int
    entity_object_types: dict[str, int] = field(default_factory=dict)


def _find_level_dirs(root: Path) -> dict[str, list[Path]]:
    """Map each level -> list of its per-day directories across all months."""
    out: dict[str, list[Path]] = defaultdict(list)
    for month_dir in sorted(root.glob("2025/*")):
        if not month_dir.is_dir():
            continue
        for level in LEVELS:
            level_dir = month_dir / level
            if level_dir.is_dir():
                day_dirs = [d for d in sorted(level_dir.glob("*")) if d.is_dir()]
                out[level].extend(day_dirs)
    return out


def _profile_csv(level: str, path: Path) -> GroupProfile:
    """Read one CSV and summarize its shape, cadence, and entities."""
    df = pd.read_csv(path, sep=";", dtype=str, low_memory=False)
    group = path.name.split(f"_{level}_")[0]

    feature_cols = [c for c in df.columns if c not in META_COLS]
    # A column is "numeric feature" if it parses as a float on a sample.
    numeric = []
    sample = df.head(200)
    for c in feature_cols:
        parsed = pd.to_numeric(sample[c], errors="coerce")
        if parsed.notna().mean() > 0.8:  # mostly numeric
            numeric.append(c)

    entities = df["fdn"].dropna().unique() if "fdn" in df.columns else []
    obj_types: Counter = Counter()
    for fdn in entities:
        comps = parse_fdn(fdn)
        # The most specific (last) managed-object key names the entity type.
        if comps:
            obj_types[list(comps.keys())[-1]] += 1

    ts_col = "rdiTimestampTZ" if "rdiTimestampTZ" in df.columns else None
    distinct_ts = df[ts_col].nunique() if ts_col else 0

    return GroupProfile(
        level=level,
        group=group,
        sample_file=str(path),
        n_rows=len(df),
        n_cols=df.shape[1],
        n_numeric_feature_cols=len(numeric),
        feature_cols_preview=numeric[:12],
        distinct_timestamps=distinct_ts,
        distinct_entities=len(entities),
        entity_object_types=dict(obj_types),
    )


def _profile_alarms(root: Path) -> dict:
    """Summarize the alarm feed that supplies labels."""
    path = root / "alarms" / "alarms.csv"
    if not path.exists():
        return {"present": False}
    df = pd.read_csv(path, sep=";", dtype=str)
    df["eventTime"] = pd.to_datetime(df["eventTime"], errors="coerce", utc=True)
    df["clearTime"] = pd.to_datetime(df["clearTime"], errors="coerce", utc=True)
    dur = (df["clearTime"] - df["eventTime"]).dt.total_seconds() / 60.0

    # Which managed-object type does each alarm attach to (from fdn)?
    obj_types: Counter = Counter()
    for fdn in df["fdn"].dropna():
        comps = parse_fdn(fdn)
        if comps:
            obj_types[list(comps.keys())[-1]] += 1

    return {
        "present": True,
        "n_alarms": len(df),
        "columns": list(df.columns),
        "probable_causes": dict(Counter(df["probableCause"]).most_common()),
        "event_types": dict(Counter(df["eventType"]).most_common()),
        "severities": dict(Counter(df["perceivedSeverity"]).most_common()),
        "attaches_to_object_types": dict(obj_types.most_common()),
        "time_min": str(df["eventTime"].min()),
        "time_max": str(df["eventTime"].max()),
        "duration_min_minutes": round(float(dur.min()), 1) if dur.notna().any() else None,
        "duration_median_minutes": round(float(dur.median()), 1) if dur.notna().any() else None,
        "duration_max_minutes": round(float(dur.max()), 1) if dur.notna().any() else None,
    }


def profile(root: Path) -> dict:
    """Build the full dataset profile dict."""
    level_days = _find_level_dirs(root)

    # Date range across all months, from infra day-dir names.
    all_days = sorted({d.name for d in level_days.get("infra", [])})
    date_range = {"first": all_days[0], "last": all_days[-1], "n_days": len(all_days)} \
        if all_days else {}

    groups: list[GroupProfile] = []
    for level in ["infra", "platform", "pm"]:  # cm profiled lightly below
        days = level_days.get(level, [])
        if not days:
            continue
        sample_day = days[len(days) // 2]  # a mid-range day, avoids edge outliers
        for csv in sorted(sample_day.glob("*.csv")):
            try:
                groups.append(_profile_csv(level, csv))
            except Exception as exc:  # noqa: BLE001 - profiling must not hard-fail
                groups.append(GroupProfile(level, csv.name, str(csv), -1, -1, -1,
                                           [f"ERROR: {exc}"], 0, 0))

    # CM: just count groups + columns (config, not time-series features).
    cm_days = level_days.get("cm", [])
    cm_groups = []
    if cm_days:
        for csv in sorted(cm_days[len(cm_days) // 2].glob("*.csv")):
            try:
                head = pd.read_csv(csv, sep=";", dtype=str, nrows=1)
                cm_groups.append({"group": csv.name.split("_cm_")[0],
                                  "n_cols": head.shape[1]})
            except Exception as exc:  # noqa: BLE001
                cm_groups.append({"group": csv.name, "error": str(exc)})

    return {
        "root": str(root),
        "date_range": date_range,
        "levels_present": {lv: len(level_days.get(lv, [])) for lv in LEVELS},
        "groups": [g.__dict__ for g in groups],
        "cm_groups": cm_groups,
        "alarms": _profile_alarms(root),
    }


def _render_markdown(p: dict) -> str:
    """Render the profile dict as a readable DATA_NOTES.md."""
    lines: list[str] = []
    w = lines.append
    w("# Dataset notes (auto-profiled)\n")
    w("Generated by `python -m autofm.profile_dataset`. Describes the real "
      "multi-level O-RAN telemetry + alarm dataset the FM pipeline trains on.\n")

    dr = p.get("date_range", {})
    if dr:
        w(f"**Date range:** {dr['first']} to {dr['last']} ({dr['n_days']} days)  ")
    lp = p["levels_present"]
    w(f"**Levels (day-dir counts):** infra={lp.get('infra', 0)}, "
      f"platform={lp.get('platform', 0)}, pm={lp.get('pm', 0)}, cm={lp.get('cm', 0)}\n")

    w("## Telemetry groups\n")
    w("| Level | Group | Rows/day | Cols | Numeric features | Distinct ROPs | Entities | Entity types |")
    w("|-------|-------|---------:|-----:|-----------------:|--------------:|---------:|--------------|")
    for g in p["groups"]:
        et = ", ".join(f"{k}:{v}" for k, v in g["entity_object_types"].items()) or "-"
        w(f"| {g['level']} | {g['group']} | {g['n_rows']} | {g['n_cols']} | "
          f"{g['n_numeric_feature_cols']} | {g['distinct_timestamps']} | "
          f"{g['distinct_entities']} | {et} |")
    w("")

    w("## CM (config) groups\n")
    w(", ".join(f"{c.get('group')}({c.get('n_cols', '?')})" for c in p["cm_groups"]) or "none")
    w("")

    a = p["alarms"]
    w("## Alarms (label source)\n")
    if not a.get("present"):
        w("_No alarms.csv found._\n")
    else:
        w(f"- **Count:** {a['n_alarms']}  ")
        w(f"- **Columns:** {', '.join(a['columns'])}  ")
        w(f"- **Time span:** {a['time_min']} to {a['time_max']}  ")
        w(f"- **Duration (min/median/max minutes):** {a['duration_min_minutes']} / "
          f"{a['duration_median_minutes']} / {a['duration_max_minutes']}  ")
        w(f"- **Severities:** {a['severities']}  ")
        w("\n**Probable causes (label candidates):**\n")
        for cause, n in a["probable_causes"].items():
            w(f"  - {cause}: {n}")
        w("\n**Alarm attaches to object types (from fdn):**\n")
        for obj, n in a["attaches_to_object_types"].items():
            w(f"  - {obj}: {n}")
    w("")

    w("## How labeling works here\n")
    w("Labels are **not** inline in the telemetry. Each alarm in `alarms.csv` is "
      "an interval `[eventTime, clearTime]` on a specific `fdn`. A telemetry "
      "window for an entity is labeled by whether an alarm is active on that "
      "entity (or an ancestor in its fdn) within the forecast horizon. This "
      "makes it a genuine *fault prediction* task, not a lookup.\n")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Profile the autofm telemetry dataset.")
    ap.add_argument("--root", required=True, help="Dataset root (autofm-dataset-ie).")
    ap.add_argument("--out", default="docs/DATA_NOTES.md", help="Markdown output path.")
    ap.add_argument("--json-out", default="artifacts/data_profile.json",
                    help="Machine-readable profile output path.")
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser()
    if not root.exists():
        raise SystemExit(f"Dataset root not found: {root}")

    prof = profile(root)

    md_path = Path(args.out)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_render_markdown(prof))

    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(prof, indent=2))

    print(f"Wrote {md_path} and {json_path}")
    print(f"Groups profiled: {len(prof['groups'])} | alarms: "
          f"{prof['alarms'].get('n_alarms', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
