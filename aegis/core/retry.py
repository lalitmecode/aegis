"""Shared notion of which failures are worth retrying.

A connection that could not be opened tells you nothing about the request. A
broker that answered and said no tells you everything. Only the first kind is
retried anywhere in this codebase.
"""

from __future__ import annotations

import socket
from time import sleep as _sleep
from typing import Any, Callable

import requests.exceptions as _requests_exc

try:  # pragma: no cover - present transitively, but not required
    import httpx as _httpx

    _HTTPX_ERRORS: tuple[type[BaseException], ...] = (
        _httpx.ConnectError,
        _httpx.ConnectTimeout,
        _httpx.ReadTimeout,
    )
except Exception:  # pragma: no cover
    _HTTPX_ERRORS = ()

#: Connection-level failures. A DNS blip is not a decision.
TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    _requests_exc.ConnectionError,
    _requests_exc.Timeout,
    socket.gaierror,
) + _HTTPX_ERRORS

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_SECONDS = 0.5


def retry_transient(
    call: Callable[[], Any],
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    sleeper: Callable[[float], None] | None = None,
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> Any:
    """Call ``call``, retrying transient connection failures with backoff.

    Anything not in :data:`TRANSIENT_ERRORS` propagates immediately.
    """
    sleeper = sleeper or _sleep
    delay = backoff_seconds
    attempts = max(1, int(max_attempts))

    for attempt in range(1, attempts + 1):
        try:
            return call()
        except TRANSIENT_ERRORS as exc:
            if on_retry is not None:
                on_retry(attempt, exc)
            if attempt == attempts:
                raise
            sleeper(delay)
            delay *= 2
