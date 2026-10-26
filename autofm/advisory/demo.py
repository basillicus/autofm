"""Advisory demo - the worked SYSTEM_DESIGN section 7 example end to end.

FM predicts a transport Packet-Loss fault on an RpUserPlaneLink under gNB
GNB001, 3 ROPs ahead. PM projects an SLA breach on the serving cell. CM proposes
candidate actions. The Analysis Bus collects all three; the Correlation Engine
combines them into ONE ranked advisory and we print what / where / when / why /
action.

Run:
    python -m autofm.advisory.demo
"""

from __future__ import annotations

from autofm.advisory.bus import AnalysisBus, Message
from autofm.advisory.cm_stub import CMStub
from autofm.advisory.correlation import CorrelationEngine
from autofm.advisory.pm_stub import PMStub


def build_fm_prediction() -> Message:
    """The FM track's fault prediction (as it would arrive from the model)."""
    return Message(
        pillar="FM", kind="fault_prediction",
        payload={
            "fault_class": "Transport",
            "probable_cause": "packet loss (pmPdcpPktLossUl)",
            "gnb": "GNB001",
            "origin_node": "RpUserPlaneLink under GNB001",
            "affected_cell": "GNB001-1",
            "horizon_rops": 3,
            "confidence": 0.82,
        },
    )


def main() -> int:
    bus = AnalysisBus()

    # 1) FM publishes its prediction.
    bus.publish(build_fm_prediction())

    # 2) PM and CM consume it and publish their own assessments.
    PMStub(bus).assess()
    CMStub(bus).propose()

    # Show the raw bus contents (what each pillar contributed).
    print("=== Analysis Bus ===")
    for m in bus.all():
        print(f"  [{m.pillar}] {m.kind}: {m.payload}")
    print()

    # 3) Correlation Engine combines all three into one ranked advisory.
    advisory = CorrelationEngine(bus).correlate()
    if advisory is None:
        print("No advisory (missing FM prediction).")
        return 1
    print(advisory.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
