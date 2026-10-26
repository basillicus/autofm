"""Advisory layer: FM / PM / CM -> Analysis Bus -> Correlation Engine.

Reproduces SYSTEM_DESIGN section 7: the three pillars each publish structured
outputs to a shared bus; the Correlation Engine consumes them and emits one
ranked, actionable suggestion (what / where / when / why / action).
"""

from autofm.advisory.bus import AnalysisBus, Message
from autofm.advisory.correlation import Advisory, CorrelationEngine
from autofm.advisory.cm_stub import CMStub
from autofm.advisory.pm_stub import PMStub

__all__ = [
    "AnalysisBus", "Message", "PMStub", "CMStub",
    "CorrelationEngine", "Advisory",
]
