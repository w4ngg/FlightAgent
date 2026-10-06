"""The only source of time for domain decisions."""

from datetime import datetime, timedelta
from typing import Protocol

from app.config import DEFAULT_START_TIME, VIETNAM_TZ


class Clock(Protocol):
    def now(self) -> datetime: ...


def require_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Clock timestamps must be timezone-aware")
    return value


class SimulatedClock:
    def __init__(self, current_time: datetime = DEFAULT_START_TIME):
        self._initial_time = require_aware(current_time)
        self._current_time = self._initial_time

    def now(self) -> datetime:
        return self._current_time

    def advance(self, seconds: int) -> datetime:
        if type(seconds) is not int or seconds < 0:
            raise ValueError("advance(seconds) requires a nonnegative integer")
        self._current_time += timedelta(seconds=seconds)
        return self._current_time

    def reset(self, current_time: datetime | None = None) -> None:
        self._current_time = require_aware(current_time or self._initial_time)


class RealClock:
    def now(self) -> datetime:
        return datetime.now(VIETNAM_TZ)
