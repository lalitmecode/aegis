"""Market session facts, backed by Alpaca's clock and calendar.

Implements the :class:`~aegis.core.risk.MarketSession` protocol the risk guard
needs for its timing rules.

``get_clock`` answers "is the market open right now", but its ``next_open`` is
the *next* session's start -- during a live session that is tomorrow. The
mandate's ``no_new_positions_within_minutes_of_open`` rule needs *this*
session's start, which only the calendar has, so the two are combined here.

Calendar times arrive as naive datetimes in US/Eastern. They are localized
explicitly; comparing a naive datetime against an aware ``now`` would raise.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from alpaca.trading.requests import GetCalendarRequest

from aegis.core.retry import (
    DEFAULT_BACKOFF_SECONDS,
    DEFAULT_MAX_ATTEMPTS,
    retry_transient,
)
from aegis.core.risk import SessionState

EASTERN = ZoneInfo("America/New_York")


class AlpacaMarketSession:
    """Market session state from Alpaca, with a per-day calendar cache."""

    def __init__(
        self,
        trading_client: Any,
        *,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
        sleeper=None,
    ) -> None:
        self._client = trading_client
        self._max_attempts = max_attempts
        self._backoff = backoff_seconds
        self._sleeper = sleeper
        self._calendar_cache: dict[date, tuple[datetime | None, datetime | None]] = {}

    def state(self, now: datetime) -> SessionState:
        """Whether the market is open, and when this session opened and closes."""
        clock = self._call(self._client.get_clock)
        session_day = now.astimezone(EASTERN).date()
        opened_at, closes_at = self._session_bounds(session_day)
        return SessionState(
            is_open=bool(clock.is_open),
            opened_at=opened_at,
            closes_at=closes_at,
        )

    def _session_bounds(self, day: date) -> tuple[datetime | None, datetime | None]:
        """This day's open/close in UTC, or ``(None, None)`` if not a trading day."""
        if day in self._calendar_cache:
            return self._calendar_cache[day]

        entries = self._call(
            lambda: self._client.get_calendar(GetCalendarRequest(start=day, end=day))
        )
        entry = next((e for e in entries or [] if e.date == day), None)
        bounds = (
            (self._to_utc(entry.open), self._to_utc(entry.close))
            if entry is not None
            else (None, None)
        )
        self._calendar_cache[day] = bounds
        return bounds

    @staticmethod
    def _to_utc(moment: datetime | None) -> datetime | None:
        """Localize a naive Eastern calendar time to UTC."""
        if moment is None:
            return None
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=EASTERN)
        return moment.astimezone(timezone.utc)

    def _call(self, fn):
        return retry_transient(
            fn,
            max_attempts=self._max_attempts,
            backoff_seconds=self._backoff,
            sleeper=self._sleeper,
        )

    def invalidate_cache(self) -> None:
        """Drop cached calendar days (e.g. after a long-running process rolls over)."""
        self._calendar_cache.clear()
