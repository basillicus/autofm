"""Analysis Bus - the shared queue the three pillars publish to.

Rather than a closed loop between FM/PM/CM, each pillar publishes a typed
``Message`` to this central bus (SYSTEM_DESIGN section 7). The Correlation
Engine consumes the bus and correlates across pillars. The bus itself is
deliberately dumb: it just stores and hands back messages, grouped by pillar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    """One pillar's structured output on the bus.

    * ``pillar``  : "FM" | "PM" | "CM".
    * ``kind``    : short type tag (e.g. "fault_prediction", "sla_risk").
    * ``payload`` : the structured data other pillars correlate on.
    """

    pillar: str
    kind: str
    payload: dict[str, Any]


@dataclass
class AnalysisBus:
    """In-memory queue of pillar messages, retrievable by pillar."""

    _messages: list[Message] = field(default_factory=list)

    def publish(self, message: Message) -> None:
        self._messages.append(message)

    def all(self) -> list[Message]:
        return list(self._messages)

    def by_pillar(self, pillar: str) -> list[Message]:
        return [m for m in self._messages if m.pillar == pillar]

    def first(self, pillar: str) -> Message | None:
        msgs = self.by_pillar(pillar)
        return msgs[0] if msgs else None

    def clear(self) -> None:
        self._messages.clear()
