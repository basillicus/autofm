"""CM pillar (rule-based stub).

Configuration Management consumes the FM fault prediction and proposes candidate
config actions, each with an estimated effectiveness and cost, then publishes
them to the bus. Phase-1 rule-based stub (SYSTEM_DESIGN section 7): a transparent
lookup from fault class to a short ranked list of remediation actions.
"""

from __future__ import annotations

from autofm.advisory.bus import AnalysisBus, Message

# Candidate actions per fault class: (action, est_effectiveness 0-1, cost tag).
_FAULT_ACTIONS: dict[str, list[tuple[str, float, str]]] = {
    "Transport": [
        ("Re-home the user-plane bearer off the degrading link", 0.90, "low"),
        ("Throttle non-priority QoS flows on the link", 0.60, "low"),
        ("Open a transport-maintenance ticket for the link", 0.40, "medium"),
    ],
    "Resource": [
        ("Migrate the NF container to a less-loaded host", 0.85, "medium"),
        ("Raise the admission-control limit temporarily", 0.55, "low"),
    ],
    "Hardware": [
        ("Fail over the cell to a redundant radio unit", 0.80, "medium"),
        ("Dispatch field maintenance for the unit", 0.50, "high"),
    ],
    "Radio": [
        ("Adjust carrier power / re-tune the sector", 0.70, "low"),
        ("Trigger a handover of users to a neighbour cell", 0.65, "low"),
    ],
    "Performance": [
        ("Rebalance PRB allocation on the cell", 0.65, "low"),
    ],
    "Software": [
        ("Schedule a rolling restart of the NF service", 0.60, "medium"),
    ],
    "Security": [
        ("Rotate the expired key / re-authenticate the node", 0.75, "low"),
    ],
    "Other": [
        ("Raise an advisory ticket for operator review", 0.30, "low"),
    ],
}


class CMStub:
    """Proposes candidate config actions from an FM fault prediction."""

    def __init__(self, bus: AnalysisBus):
        self.bus = bus

    def propose(self) -> Message | None:
        fm = self.bus.first("FM")
        if fm is None:
            return None
        fault_class = fm.payload.get("fault_class", "Other")
        actions = _FAULT_ACTIONS.get(fault_class, _FAULT_ACTIONS["Other"])
        candidates = [
            {"action": a, "effectiveness": eff, "cost": cost}
            for (a, eff, cost) in actions
        ]
        msg = Message(
            pillar="CM", kind="candidate_actions",
            payload={"gnb": fm.payload.get("gnb"), "candidates": candidates},
        )
        self.bus.publish(msg)
        return msg
