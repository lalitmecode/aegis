"""Console tests. Every dependency is injected; nothing reaches Alpaca."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from aegis.core.audit import HashChainAudit, read_runs
from aegis.core.proposal import PortfolioState
from aegis.core.risk import SessionState
from aegis.web.app import create_app

NOW = datetime(2026, 8, 21, 15, 0, tzinfo=timezone.utc)


class StubClients:
    def __init__(self, fail=False):
        def account():
            if fail:
                raise ConnectionError("name resolution failed")
            return SimpleNamespace(
                account_number="PA000", status="AccountStatus.ACTIVE",
                equity="99859.78", buying_power="395195.12", options_trading_level=3,
            )

        self.trading = SimpleNamespace(get_account=account)


class StubPortfolio:
    def fetch(self):
        return PortfolioState(
            equity=Decimal("99859.78"), buying_power=Decimal("395195.12"),
            open_positions=3, positions_by_symbol={"AAPL": 1, "MSFT": 1, "SPY": 1},
            portfolio_delta=17.9, capital_at_risk_pct=1.5, fetched_at=NOW,
        )


class StubSession:
    def state(self, now):
        return SessionState(is_open=True, opened_at=NOW)


def client(tmp_path=None, **kw) -> TestClient:
    kw.setdefault("clients", StubClients())
    kw.setdefault("portfolio", StubPortfolio())
    kw.setdefault("session", StubSession())
    if tmp_path is not None:
        kw.setdefault("logs_dir", tmp_path)
    return TestClient(create_app(**kw))


def write_log(tmp_path, runs=1, entries=3, tamper=False):
    path = tmp_path / "audit-20260821.jsonl"
    for _ in range(runs):
        audit = HashChainAudit(path)
        for i in range(entries):
            audit.record("PROPOSED" if i == 0 else "SUBMITTED", {"n": i})
    if tamper:
        lines = path.read_text().splitlines()
        first = json.loads(lines[0])
        first["payload"] = {"n": "edited after the fact"}
        lines[0] = json.dumps(first)
        path.write_text("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------


def test_state_reports_account_market_and_portfolio():
    body = client().get("/api/state").json()
    assert body["account"]["number"] == "PA000"
    assert body["account"]["options_level"] == 3
    assert body["market"]["is_open"] is True
    assert body["portfolio"]["open_positions"] == 3
    assert body["portfolio"]["symbols"] == ["AAPL", "MSFT", "SPY"]


def test_state_degrades_to_503_when_alpaca_is_unreachable():
    """A blank 500 tells the operator nothing; 503 with the cause tells them why."""
    response = client(clients=StubClients(fail=True)).get("/api/state")
    assert response.status_code == 503
    assert "name resolution failed" in response.json()["detail"]


# --------------------------------------------------------------------------
# mandate
# --------------------------------------------------------------------------


def test_mandate_exposes_the_limits_the_guard_enforces():
    body = client().get("/api/mandate").json()
    assert body["risk_limits"]["max_loss_per_trade_usd"] == 500
    assert body["strategy"]["delta_limits"]["short_leg_abs_delta_max"] == 0.30
    assert body["universe"]["min_open_interest"] == 500
    assert body["account_type"] == "paper"


# --------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------


def test_audit_returns_verified_runs(tmp_path):
    write_log(tmp_path, runs=2, entries=4)
    body = client(tmp_path).get("/api/audit?day=20260821").json()
    assert len(body["runs"]) == 2, "each process starts a fresh chain"
    assert all(run["intact"] for run in body["runs"])
    assert all(len(run["entries"]) == 4 for run in body["runs"])


def test_audit_reports_a_tampered_entry(tmp_path):
    """Verification is recomputed on read, not taken from the file."""
    write_log(tmp_path, entries=3, tamper=True)
    run = client(tmp_path).get("/api/audit?day=20260821").json()["runs"][0]
    assert run["intact"] is False
    assert "digest does not match" in run["problem"]


def test_audit_is_empty_for_a_day_with_no_log(tmp_path):
    assert client(tmp_path).get("/api/audit?day=19990101").json()["runs"] == []


def test_days_lists_available_logs_newest_first(tmp_path):
    for stamp in ("20260819", "20260821", "20260820"):
        (tmp_path / f"audit-{stamp}.jsonl").write_text("")
    assert client(tmp_path).get("/api/days").json()["days"] == [
        "20260821", "20260820", "20260819"
    ]


# --------------------------------------------------------------------------
# the console is read-only
# --------------------------------------------------------------------------


def test_the_console_serves_the_page():
    response = client().get("/")
    assert response.status_code == 200
    assert "governance console" in response.text


def test_there_is_no_write_endpoint():
    """A second path to the broker would make the one-gate claim untrue."""
    app = create_app(clients=StubClients(), portfolio=StubPortfolio(), session=StubSession())
    methods = {m for route in app.routes for m in getattr(route, "methods", set())}
    assert methods <= {"GET", "HEAD"}, f"unexpected write methods: {methods}"


@pytest.mark.parametrize("path", ["/api/state", "/api/mandate", "/api/audit"])
def test_write_verbs_are_rejected(path):
    assert client().post(path).status_code == 405
