"""Ordered local evidence, including rejected and exceptional tool attempts."""

from copy import deepcopy
from typing import Any

from pydantic import AwareDatetime

from app.clock import Clock
from app.models import DomainModel, PositiveInt, TransactionContext


class TraceEvent(DomainModel):
    sequence_number: PositiveInt
    timestamp: AwareDatetime
    event_type: str
    context: TransactionContext
    tool_name: str | None = None
    attempt_id: str | None = None
    allowed: bool | None = None
    payload: dict[str, Any]


class TraceCollector:
    def __init__(self, clock: Clock):
        self.clock = clock
        self._events: list[TraceEvent] = []

    def record(self, event_type: str, context: TransactionContext, *, tool_name=None,
               attempt_id=None, allowed=None, **payload) -> TraceEvent:
        event = TraceEvent(sequence_number=len(self._events) + 1, timestamp=self.clock.now(),
                           event_type=event_type, context=context, tool_name=tool_name,
                           attempt_id=attempt_id, allowed=allowed, payload=deepcopy(payload))
        self._events.append(event)
        return event.model_copy(deep=True)

    def get_events(self) -> list[TraceEvent]:
        return [event.model_copy(deep=True) for event in self._events]

    def get_trace(self) -> list[dict]:
        return [event.model_dump(mode="json") for event in self._events]

