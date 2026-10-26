"""Correlation Engine - combine FM + PM + CM into one ranked advisory.

Consumes the three pillars' messages off the Analysis Bus and correlates them
(SYSTEM_DESIGN section 7): it links the FM fault prediction to the PM SLA-breach
risk it drives and to the CM action that best resolves both, then emits a single
ranked advisory answering **what / where / when / why / action**, with a
combined confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

from autofm.advisory.bus import AnalysisBus


@dataclass
class Advisory:
    """The unified, ranked suggestion delivered to the operator."""

    what: str
    where: str
    when: str
    why: str
    action: str
    confidence: float

    def render(self) -> str:
        return (
            "autofm advisory\n"
            "----------------\n"
            f"  what      : {self.what}\n"
            f"  where     : {self.where}\n"
            f"  when      : {self.when}\n"
            f"  why       : {self.why}\n"
            f"  action    : {self.action}\n"
            f"  confidence: {self.confidence:.2f}\n"
        )


class CorrelationEngine:
    """Correlates bus messages across pillars into one ranked advisory."""

    def __init__(self, bus: AnalysisBus):
        self.bus = bus

    def correlate(self) -> Advisory | None:
        fm = self.bus.first("FM")
        pm = self.bus.first("PM")
        cm = self.bus.first("CM")
        if fm is None:
            return None

        f = fm.payload
        gnb = f.get("gnb", "?")
        origin = f.get("origin_node", gnb)

        # ---- what ----
        what = (f"Predicted {f.get('fault_class', 'fault')} fault "
                f"({f.get('probable_cause', 'n/a')})")

        # ---- where ----
        cell = (pm.payload.get("cell") if pm else None) or f.get("affected_cell", "?")
        where = f"gNB {gnb}, origin {origin}; serving cell {cell}"

        # ---- when ----
        lead = f.get("horizon_rops", "?")
        when = f"fault in ~{lead} ROPs"
        if pm:
            when += f"; SLA breach risk in ~{pm.payload.get('breach_in_rops', '?')} ROPs"

        # ---- why ----
        why = (f"{f.get('fault_class', 'fault')} on {origin} propagates to the "
               f"serving cell's user plane")
        if pm:
            why += (f"; PM projects SLA risk "
                    f"{pm.payload.get('sla_breach_risk', 0):.0%} on {cell}")

        # ---- action (pick the highest-effectiveness CM candidate) ----
        best = None
        if cm and cm.payload.get("candidates"):
            best = max(cm.payload["candidates"], key=lambda c: c["effectiveness"])
        action = best["action"] if best else "Raise an advisory ticket for review"

        # ---- combined confidence ----
        # FM prediction confidence x CM action effectiveness, nudged by SLA risk.
        fm_conf = float(f.get("confidence", 0.5))
        act_eff = float(best["effectiveness"]) if best else 0.3
        sla_risk = float(pm.payload.get("sla_breach_risk", 0.5)) if pm else 0.5
        confidence = round(fm_conf * act_eff * (0.5 + 0.5 * sla_risk), 2)

        return Advisory(what, where, when, why, action, confidence)
