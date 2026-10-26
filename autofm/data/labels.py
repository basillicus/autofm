"""Alarm-to-label layer.

Telemetry carries no inline fault label. Labels come from ``alarms.csv``: each
alarm is an interval ``[eventTime, clearTime]`` on an ``fdn``. We turn that feed
into a supervised **fault-prediction** target on the cell-centric telemetry
frame produced by :mod:`autofm.data.loader`.

Two steps:

1. **Resolve scope.** Map each alarm's ``fdn`` to the cells it affects, using
   the same topology the loader uses:

   * ``NRCellDU``        -> that exact cell (``gnb`` + ``cell`` id).
   * ``S1ULink``         -> all cells of the alarm's gNB (``MeContext``).
   * ``ManagedNFService``-> all cells of the alarm's gNB (service name prefix).
   * ``OCloudNode``      -> all cells in the alarm's region (SubNetwork).

2. **Forecast label.** A cell at ROP ``t`` is **positive** if any alarm is
   active on it within the next ``horizon`` ROPs (``t`` exclusive ..
   ``t + horizon`` inclusive). This makes the task genuine forecasting - the
   model must see the precursor, not the alarm itself. We emit both a binary
   ``will_fault`` label and a multi-class ``fault_class`` (grouped probable
   cause; ``0`` = Normal).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from autofm.data.loader import _region_of
from autofm.profile_dataset import parse_fdn

ROP = pd.Timedelta(minutes=15)

# Group the 24 raw probableCause values into a compact class taxonomy. This
# keeps the first classifier tractable and maps causes to the level they most
# naturally belong to. 0 is reserved for Normal (no imminent alarm).
CAUSE_TO_CLASS: dict[str, str] = {
    # compute / host pressure
    "congestion": "Resource",
    "outOfMemory": "Resource",
    "storageCapacityProblem": "Resource",
    "highTemperature": "Hardware",
    "coolingSystemFailure": "Hardware",
    "powerProblem": "Hardware",
    "equipmentMalfunction": "Hardware",
    "replaceableUnitProblem": "Hardware",
    # radio / signal
    "degradedSignal": "Radio",
    "lossOfSignal": "Radio",
    "lossOfFrame": "Radio",
    "lossOfSynchronisation": "Radio",
    "performanceDegraded": "Performance",
    "thresholdCrossed": "Performance",
    # transport / connectivity
    "lanError": "Transport",
    "connectionEstablishmentError": "Transport",
    # software / config / security
    "softwareError": "Software",
    "configurationError": "Software",
    "proceduralError": "Software",
    "informationModificationDetected": "Security",
    "authenticationFailure": "Security",
    "keyExpired": "Security",
    "cableTamper": "Security",
    "indeterminate": "Other",
}

CLASS_NAMES = ["Normal", "Resource", "Hardware", "Radio", "Performance",
               "Transport", "Software", "Security", "Other"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


@dataclass
class ResolvedAlarm:
    """An alarm mapped to a cell scope + a time interval + a class id."""

    scope_type: str           # "cell" | "gnb" | "region"
    gnb: str | None
    cell: str | None
    region: str | None
    start: pd.Timestamp
    end: pd.Timestamp
    class_id: int


def _resolve_scope(fdn: str) -> tuple[str, str | None, str | None, str | None]:
    """Return (scope_type, gnb, cell, region) for an alarm fdn."""
    comps = parse_fdn(fdn)
    obj = list(comps.keys())[-1] if comps else None
    region = _region_of(fdn)
    if obj == "NRCellDU":
        return "cell", comps.get("MeContext"), comps.get("NRCellDU"), region
    if obj == "S1ULink":
        return "gnb", comps.get("MeContext"), None, region
    if obj == "ManagedNFService":
        svc = comps.get("ManagedNFService", "")
        return "gnb", svc.split("-", 1)[0] or None, None, region
    if obj == "OCloudNode":
        return "region", None, None, region
    return "region", None, None, region  # default: widest scope


def load_alarms(alarms_csv: str) -> list[ResolvedAlarm]:
    """Read and resolve every alarm into a cell-scoped interval + class."""
    a = pd.read_csv(alarms_csv, sep=";", dtype=str)
    a["eventTime"] = pd.to_datetime(a["eventTime"], utc=True, errors="coerce")
    a["clearTime"] = pd.to_datetime(a["clearTime"], utc=True, errors="coerce")
    # An alarm with no clear time is treated as open until end of its day + 1d.
    a["clearTime"] = a["clearTime"].fillna(a["eventTime"] + pd.Timedelta(days=1))

    resolved: list[ResolvedAlarm] = []
    for _, r in a.iterrows():
        if pd.isna(r["eventTime"]):
            continue
        st, gnb, cell, region = _resolve_scope(r["fdn"])
        cls = CAUSE_TO_CLASS.get(r.get("probableCause"), "Other")
        resolved.append(ResolvedAlarm(
            scope_type=st, gnb=gnb, cell=cell, region=region,
            start=r["eventTime"], end=r["clearTime"],
            class_id=CLASS_TO_ID[cls],
        ))
    return resolved


def label_frame(
    frame: pd.DataFrame,
    alarms: list[ResolvedAlarm],
    *,
    horizon: int = 5,
    ts_col: str = "rdiTimestampTZ",
) -> pd.DataFrame:
    """Add ``will_fault`` (0/1) and ``fault_class`` (int) to the telemetry frame.

    A row (gnb, cell, t) is positive if an alarm in its scope is active anywhere
    in ``(t, t + horizon*ROP]``. When several alarms qualify, the earliest one's
    class wins (the imminent fault). Normal rows get class 0.

    Args:
        frame:   cell-centric telemetry frame (must have gnb, cell, region, ts).
        alarms:  resolved alarms from :func:`load_alarms`.
        horizon: forecast horizon in ROPs (15-min steps).
    """
    frame = frame.copy()
    # Both telemetry and alarms are UTC. Drop tz to compare as plain datetime64
    # (numpy comparisons below are tz-naive); this is lossless since all UTC.
    ts_naive = frame[ts_col].dt.tz_convert("UTC").dt.tz_localize(None)
    t = ts_naive.to_numpy()
    will = np.zeros(len(frame), dtype=np.int8)
    cls = np.zeros(len(frame), dtype=np.int64)
    # Earliest alarm start seen per row, so the nearest fault's class wins.
    best_start = np.full(len(frame), np.datetime64("2100-01-01T00:00:00"), dtype="datetime64[ns]")

    horizon_td = horizon * ROP

    # Pre-index rows by scope key for cheap matching.
    gnb_arr = frame["gnb"].to_numpy()
    cell_arr = frame["cell"].to_numpy()
    region_arr = frame["region"].to_numpy()

    # Forecast window end per row (tz-naive, matching ``t``).
    win_end = (ts_naive + horizon_td).to_numpy()

    for al in alarms:
        # Row matches the alarm's scope?
        if al.scope_type == "cell":
            mask = (gnb_arr == al.gnb) & (cell_arr == al.cell)
        elif al.scope_type == "gnb":
            mask = gnb_arr == al.gnb
        else:  # region
            mask = region_arr == al.region
        if not mask.any():
            continue

        # Alarm overlaps the forecast window (t, t+horizon]:
        #   alarm active in window  <=>  start <= win_end AND end > t
        start = np.datetime64(al.start.to_datetime64())
        end = np.datetime64(al.end.to_datetime64())
        overlap = mask & (start <= win_end) & (end > t)
        if not overlap.any():
            continue

        will[overlap] = 1
        # Keep the class of the earliest-starting qualifying alarm.
        closer = overlap & (start < best_start)
        best_start[closer] = start
        cls[closer] = al.class_id

    frame["will_fault"] = will
    frame["fault_class"] = cls
    return frame
