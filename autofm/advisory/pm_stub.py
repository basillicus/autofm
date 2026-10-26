"""PM pillar (rule-based stub).

Performance Management consumes an FM fault prediction and estimates the KPI
impact + SLA breach risk it implies, then publishes that to the bus. This is a
Phase-1 rule-based stub (SYSTEM_DESIGN section 7): no model, just a transparent
mapping from fault type/severity to an SLA risk and a projected breach horizon.
"""

from __future__ import annotations

from autofm.advisory.bus import AnalysisBus, Message

# How much each fault class threatens the SLA, and how soon it tends to bite
# (in ROPs after the fault onset). Rule-based, tunable, transparent.
_FAULT_SLA_RISK: dict[str, tuple[float, int]] = {
    "Transport": (0.85, 3),       # packet loss hits user-plane SLA fast
    "Radio": (0.70, 4),
    "Performance": (0.65, 5),
    "Resource": (0.60, 6),
    "Hardware": (0.55, 6),
    "Software": (0.40, 8),
    "Security": (0.35, 8),
    "Other": (0.30, 10),
}


class PMStub:
    """Estimates SLA breach risk from an FM fault prediction."""

    def __init__(self, bus: AnalysisBus):
        self.bus = bus

    def assess(self) -> Message | None:
        """Read the FM prediction off the bus, publish an SLA-risk message."""
        fm = self.bus.first("FM")
        if fm is None:
            return None
        fault = fm.payload
        fault_class = fault.get("fault_class", "Other")
        risk, breach_in = _FAULT_SLA_RISK.get(fault_class, (0.3, 10))

        # Serving cell under the faulting gNB takes the SLA hit.
        cell = fault.get("affected_cell") or f"{fault.get('gnb', '?')}-1"
        msg = Message(
            pillar="PM", kind="sla_risk",
            payload={
                "cell": cell,
                "gnb": fault.get("gnb"),
                "sla_breach_risk": round(risk, 2),
                "breach_in_rops": breach_in,
                "kpi": "user-plane throughput / packet loss",
            },
        )
        self.bus.publish(msg)
        return msg
