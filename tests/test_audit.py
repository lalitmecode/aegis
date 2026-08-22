"""The audit chain's job is to make silent edits impossible."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from aegis.core.audit import GENESIS, HashChainAudit

T0 = datetime(2026, 8, 19, 15, 0, tzinfo=timezone.utc)


def build(tmp_path=None):
    ticks = iter(T0 + timedelta(seconds=i) for i in range(100))
    return HashChainAudit(tmp_path, clock=lambda: next(ticks))


def test_a_fresh_chain_verifies():
    audit = build()
    assert audit.verify() == (True, None)
    assert audit.head == GENESIS


def test_entries_link_to_their_predecessor():
    audit = build()
    first = audit.record("REJECTED", {"proposal_id": "p1"})
    second = audit.record("SUBMITTED", {"proposal_id": "p2"})

    assert audit.entries[0].previous == GENESIS
    assert audit.entries[1].previous == first
    assert audit.head == second
    assert audit.verify() == (True, None)


def test_editing_a_payload_breaks_the_chain():
    audit = build()
    audit.record("REJECTED", {"reason": "max loss exceeded"})
    audit.record("SUBMITTED", {"broker_order_id": "ord-1"})

    # Rewrite history: make the refusal look like it never happened.
    audit._entries[0] = replace(audit._entries[0], payload={"reason": "all good"})

    intact, problem = audit.verify()
    assert not intact
    assert "entry 0 digest" in problem


def test_deleting_an_entry_breaks_the_chain():
    audit = build()
    audit.record("REJECTED", {"n": 1})
    audit.record("SUBMITTED", {"n": 2})
    audit.record("SUBMITTED", {"n": 3})

    del audit._entries[1]

    intact, problem = audit.verify()
    assert not intact
    assert problem is not None


def test_it_satisfies_the_gateway_audit_protocol(tmp_path):
    audit = build(tmp_path / "audit.jsonl")
    digest = audit.record("SUBMITTED", {"broker_order_id": "ord-9"})
    assert isinstance(digest, str) and len(digest) == 64
    assert audit.events() == ["SUBMITTED"]


def test_entries_are_persisted_as_jsonl(tmp_path):
    path = tmp_path / "nested" / "audit.jsonl"
    audit = build(path)
    audit.record("REJECTED", {"proposal_id": "p1"})
    audit.record("SUBMITTED", {"proposal_id": "p1"})

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    written = [json.loads(line) for line in lines]
    assert written[0]["event"] == "REJECTED"
    assert written[1]["previous"] == written[0]["digest"]


def test_non_serialisable_payloads_do_not_crash_the_log(tmp_path):
    from decimal import Decimal

    audit = build(tmp_path / "audit.jsonl")
    audit.record("SUBMITTED", {"max_loss": Decimal("390.00"), "at": T0})
    assert audit.verify() == (True, None)


# --------------------------------------------------------------------------
# reading a log back
# --------------------------------------------------------------------------


def test_read_runs_splits_a_file_into_its_processes(tmp_path):
    """Each run restarts from genesis, so a day's file holds several chains."""
    from aegis.core.audit import read_runs

    path = tmp_path / "audit.jsonl"
    for _ in range(3):
        audit = build(path)
        audit.record("PROPOSED", {})
        audit.record("SUBMITTED", {})

    runs = read_runs(path)
    assert len(runs) == 3
    assert all(run.intact for run in runs)
    assert all(len(run.entries) == 2 for run in runs)


def test_read_runs_detects_an_edited_entry(tmp_path):
    import json

    from aegis.core.audit import read_runs

    path = tmp_path / "audit.jsonl"
    audit = build(path)
    audit.record("REJECTED", {"reason": "max loss exceeded"})
    audit.record("SUBMITTED", {"broker_order_id": "ord-1"})

    lines = path.read_text().splitlines()
    edited = json.loads(lines[0])
    edited["payload"] = {"reason": "all good"}
    lines[0] = json.dumps(edited)
    path.write_text("\n".join(lines) + "\n")

    run = read_runs(path)[0]
    assert not run.intact
    assert "digest does not match" in run.problem


def test_read_runs_on_a_missing_file_is_empty(tmp_path):
    from aegis.core.audit import read_runs

    assert read_runs(tmp_path / "nope.jsonl") == []
