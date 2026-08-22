"""Append-only, hash-chained decision log.

The mandate requires every decision to be logged. A plain log satisfies that
only if nobody can quietly rewrite it afterwards, so each entry carries the
hash of the entry before it. Altering or removing any entry changes its hash
and breaks every link downstream, which :meth:`HashChainAudit.verify` detects.

This is tamper-*evident*, not tamper-proof: anyone who can rewrite the file can
also recompute the chain. It makes silent edits impossible, not impossible
edits. Anchoring the head hash somewhere the writer does not control is what
would close that gap.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

GENESIS = "0" * 64


@dataclass(frozen=True, slots=True)
class AuditEntry:
    """One recorded decision, linked to its predecessor."""

    seq: int
    event: str
    payload: Mapping[str, Any]
    recorded_at: str
    previous: str
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "event": self.event,
            "payload": dict(self.payload),
            "recorded_at": self.recorded_at,
            "previous": self.previous,
            "digest": self.digest,
        }


def _digest(seq: int, event: str, payload: Mapping[str, Any], recorded_at: str, previous: str) -> str:
    body = json.dumps(
        {
            "seq": seq,
            "event": event,
            "payload": payload,
            "recorded_at": recorded_at,
            "previous": previous,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class AuditRun:
    """One process's worth of entries, and whether the chain holds."""

    entries: tuple[AuditEntry, ...]
    intact: bool
    problem: str | None = None

    @property
    def head(self) -> str:
        return self.entries[-1].digest if self.entries else GENESIS


def _verify(entries) -> tuple[bool, str | None]:
    previous = GENESIS
    for i, entry in enumerate(entries):
        if entry.seq != i:
            return False, f"entry {i} claims seq {entry.seq}"
        if entry.previous != previous:
            return False, f"entry {i} does not link to entry {i - 1}"
        expected = _digest(
            entry.seq, entry.event, entry.payload, entry.recorded_at, entry.previous
        )
        if entry.digest != expected:
            return False, f"entry {i} digest does not match its contents"
        previous = entry.digest
    return True, None


def read_runs(path: str | os.PathLike[str]) -> list[AuditRun]:
    """Read a log file back as its constituent runs, each verified.

    Each process starts a fresh chain from genesis and appends to the day's
    file, so one file holds several chains rather than one. Splitting on
    ``seq == 0`` recovers them; verifying the file as a single chain would
    report a break at every process boundary and mean nothing.
    """
    file = Path(path)
    if not file.exists():
        return []

    runs: list[list[AuditEntry]] = []
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        entry = AuditEntry(
            seq=raw["seq"],
            event=raw["event"],
            payload=raw.get("payload") or {},
            recorded_at=raw["recorded_at"],
            previous=raw["previous"],
            digest=raw["digest"],
        )
        if entry.seq == 0 or not runs:
            runs.append([])
        runs[-1].append(entry)

    out = []
    for entries in runs:
        intact, problem = _verify(entries)
        out.append(AuditRun(tuple(entries), intact, problem))
    return out


class HashChainAudit:
    """Implements the gateway's ``AuditChain`` protocol, with verification."""

    def __init__(self, path: str | os.PathLike[str] | None = None, *, clock=None) -> None:
        self._entries: list[AuditEntry] = []
        self._path = Path(path) if path else None
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: str, payload: Mapping[str, Any]) -> str:
        """Append an entry and return its digest."""
        seq = len(self._entries)
        previous = self._entries[-1].digest if self._entries else GENESIS
        recorded_at = self._clock().isoformat()
        payload = dict(payload)
        entry = AuditEntry(
            seq=seq,
            event=event,
            payload=payload,
            recorded_at=recorded_at,
            previous=previous,
            digest=_digest(seq, event, payload, recorded_at, previous),
        )
        self._entries.append(entry)

        if self._path is not None:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry.to_dict(), default=str) + "\n")
        return entry.digest

    def verify(self) -> tuple[bool, str | None]:
        """Recompute the chain. Returns ``(intact, first_problem)``."""
        return _verify(self._entries)

    @property
    def entries(self) -> tuple[AuditEntry, ...]:
        return tuple(self._entries)

    @property
    def head(self) -> str:
        return self._entries[-1].digest if self._entries else GENESIS

    def events(self) -> list[str]:
        return [entry.event for entry in self._entries]

    def __len__(self) -> int:
        return len(self._entries)
