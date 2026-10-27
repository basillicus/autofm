"""Shared FDN (Full Distinguished Name) parsing utilities.

The dataset identifies every managed object by an ``fdn`` string, a comma-
separated chain of ``key=value`` components, e.g.::

    SubNetwork=IE,SubNetwork=RAN,SubNetwork=COR,MeContext=CORDU08,...,NRCellDU=CIDU1U

Several layers (the loader, the labeler, and the profiler) need to parse these
identically, so the parsing lives here as the single source of truth rather
than in any one consumer. Keeping it dependency-free (stdlib ``re`` only) means
the data layer never has to import a CLI/profiling module just to read an fdn.
"""

from __future__ import annotations

import re

# Columns every telemetry CSV carries (trailing metadata block, ;-delimited).
META_COLS = ["fdn", "mecontext", "cell", "rdiTimestamp", "rdiTimestampTZ", "ropCount"]

# Managed-object levels, outermost first. The topology the whole project uses.
LEVELS = ["infra", "platform", "pm", "cm"]

# One ``key=value`` component of an fdn. Values run until the next comma.
_FDN_KV = re.compile(r"(\w+)=([^,]+)")

# The region lives at the THIRD SubNetwork level: SubNetwork=<NET>,
# SubNetwork=<RAN>, SubNetwork=<REGION>, ... (see docs/DATASET_SPEC.md). The top
# two SubNetwork levels are fixed infrastructure, not regions.
_REGION_SUBNETWORK_INDEX = 2


def parse_fdn(fdn: str) -> dict[str, str]:
    """Return the ``key=value`` components of an fdn as a dict (last wins).

    Note that ``SubNetwork`` repeats in an fdn; a dict keeps only the last one,
    so use :func:`fdn_kv_pairs` when the repeated levels matter (e.g. region).
    """
    return {k: v for k, v in _FDN_KV.findall(fdn or "")}


def fdn_kv_pairs(fdn: str) -> list[tuple[str, str]]:
    """Return all ``key=value`` pairs in order (keeps repeated keys)."""
    return _FDN_KV.findall(fdn or "")


def region_of(fdn: str) -> str | None:
    """Region for an fdn = its third ``SubNetwork`` level, else ``None``.

    ``SubNetwork=IE,SubNetwork=RAN,SubNetwork=COR,...`` -> ``COR``. Used to link
    cells to the host pool serving their region, since host names (``CORDU01``)
    do not match a cell's ManagedElement 1:1 - hosts are a regional O-Cloud
    pool, not per-gNB. An fdn with fewer than three SubNetwork levels has no
    region and yields ``None`` (rather than mislabeling ``RAN`` as a region).
    """
    subs = [v for k, v in fdn_kv_pairs(fdn) if k == "SubNetwork"]
    if len(subs) > _REGION_SUBNETWORK_INDEX:
        return subs[_REGION_SUBNETWORK_INDEX]
    return None


def object_type(fdn: str) -> str | None:
    """The most specific (last) managed-object key names the entity type."""
    comps = parse_fdn(fdn)
    if not comps:
        return None
    return list(comps.keys())[-1]
