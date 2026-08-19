"""Session tests. The Alpaca client is a fake -- no live calls."""

from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest
import requests.exceptions
from alpaca.trading.models import Calendar

from aegis.core.session import AlpacaMarketSession

# 2026-08-19 14:00 UTC == 10:00 ET, mid-session.
NOW = datetime(2026, 8, 19, 14, 0, tzinfo=timezone.utc)
TRADING_DAY = date(2026, 8, 19)


def calendar_entry(day=TRADING_DAY, open_="09:30", close="16:00"):
    return Calendar(date=day.isoformat(), open=open_, close=close)


class FakeClient:
    def __init__(self, is_open=True, entries=None, fail_times=0):
        self.is_open = is_open
        self.entries = [calendar_entry()] if entries is None else entries
        self.fail_times = fail_times
        self.clock_calls = 0
        self.calendar_calls = 0

    def get_clock(self):
        self.clock_calls += 1
        if self.clock_calls <= self.fail_times:
            raise requests.exceptions.ConnectionError("Temporary failure in name resolution")
        return SimpleNamespace(is_open=self.is_open)

    def get_calendar(self, filters=None):
        self.calendar_calls += 1
        return self.entries


def build(client, **kw):
    return AlpacaMarketSession(client, sleeper=lambda _s: None, **kw)


def test_is_open_comes_from_the_clock():
    assert build(FakeClient(is_open=True)).state(NOW).is_open is True
    assert build(FakeClient(is_open=False)).state(NOW).is_open is False


def test_opened_at_is_this_sessions_open_localized_to_utc():
    """09:30 ET on a summer date is 13:30 UTC, not 09:30 UTC."""
    state = build(FakeClient()).state(NOW)
    assert state.opened_at == datetime(2026, 8, 19, 13, 30, tzinfo=timezone.utc)
    assert state.closes_at == datetime(2026, 8, 19, 20, 0, tzinfo=timezone.utc)


def test_opened_at_is_comparable_with_an_aware_now():
    """A naive calendar datetime would raise here."""
    state = build(FakeClient()).state(NOW)
    assert state.opened_at < NOW  # would TypeError if left naive


def test_a_half_day_close_is_respected():
    client = FakeClient(entries=[calendar_entry(close="13:00")])
    assert build(client).state(NOW).closes_at == datetime(
        2026, 8, 19, 17, 0, tzinfo=timezone.utc
    )


def test_a_non_trading_day_has_no_session_bounds():
    state = build(FakeClient(is_open=False, entries=[])).state(NOW)
    assert state.is_open is False
    assert state.opened_at is None
    assert state.closes_at is None


def test_the_calendar_is_cached_per_day():
    client = FakeClient()
    session = build(client)
    for _ in range(4):
        session.state(NOW)

    assert client.calendar_calls == 1, "calendar should be fetched once per day"
    assert client.clock_calls == 4, "but is_open must stay live"


def test_cache_can_be_invalidated():
    client = FakeClient()
    session = build(client)
    session.state(NOW)
    session.invalidate_cache()
    session.state(NOW)
    assert client.calendar_calls == 2


def test_transient_connection_errors_are_retried():
    client = FakeClient(fail_times=2)
    assert build(client).state(NOW).is_open is True
    assert client.clock_calls == 3


def test_persistent_connection_failure_propagates():
    client = FakeClient(fail_times=99)
    with pytest.raises(requests.exceptions.ConnectionError):
        build(client).state(NOW)
    assert client.clock_calls == 3


def test_backoff_is_exponential():
    delays: list[float] = []
    client = FakeClient(fail_times=2)
    AlpacaMarketSession(client, sleeper=delays.append, backoff_seconds=0.5).state(NOW)
    assert delays == [0.5, 1.0]


def test_it_satisfies_the_market_session_protocol_the_guard_uses():
    """End to end: the real guard drives this session through its public API."""
    import pathlib
    from decimal import Decimal

    import yaml
    from alpaca.trading.enums import OrderSide, PositionIntent

    from aegis.core.proposal import OrderLeg, PortfolioState, TradeProposal
    from aegis.core.risk import RiskGuard

    mandate = yaml.safe_load(
        (pathlib.Path(__file__).resolve().parent.parent / "config" / "mandate.yaml").read_text()
    )
    proposal = TradeProposal(
        proposal_id="prop-001",
        underlying="SPY",
        structure="vertical_credit_spread",
        legs=(
            OrderLeg("SPY260918P00753000", OrderSide.SELL, PositionIntent.SELL_TO_OPEN,
                     delta=Decimal("-0.30")),
            OrderLeg("SPY260918P00748000", OrderSide.BUY, PositionIntent.BUY_TO_OPEN,
                     delta=Decimal("-0.22")),
        ),
        quantity=1,
        limit_price=Decimal("1.10"),
        max_loss_usd=Decimal("390"),
    )
    state = PortfolioState(
        equity=Decimal("100000"),
        buying_power=Decimal("400000"),
        open_positions=0,
        positions_by_symbol={},
        portfolio_delta=0.0,
        capital_at_risk_pct=0.0,
        fetched_at=NOW,
    )

    guard = RiskGuard(mandate, session=build(FakeClient()), clock=lambda: NOW)
    decision = guard.evaluate(proposal, state)
    assert decision.approved, decision.reasons


def test_a_closed_market_reaches_the_guard_as_a_refusal():
    import pathlib
    from decimal import Decimal

    import yaml
    from alpaca.trading.enums import OrderSide, PositionIntent

    from aegis.core.proposal import OrderLeg, PortfolioState, TradeProposal
    from aegis.core.risk import RiskGuard

    mandate = yaml.safe_load(
        (pathlib.Path(__file__).resolve().parent.parent / "config" / "mandate.yaml").read_text()
    )
    proposal = TradeProposal(
        proposal_id="p", underlying="SPY", structure="vertical_credit_spread",
        legs=(OrderLeg("SPY260918P00753000", OrderSide.SELL, PositionIntent.SELL_TO_OPEN,
                       delta=Decimal("-0.30")),
              OrderLeg("SPY260918P00748000", OrderSide.BUY, PositionIntent.BUY_TO_OPEN,
                       delta=Decimal("-0.22"))),
        quantity=1, limit_price=Decimal("1.10"), max_loss_usd=Decimal("390"),
    )
    state = PortfolioState(
        equity=Decimal("100000"), buying_power=Decimal("400000"), open_positions=0,
        positions_by_symbol={}, portfolio_delta=0.0, capital_at_risk_pct=0.0, fetched_at=NOW,
    )
    guard = RiskGuard(mandate, session=build(FakeClient(is_open=False)), clock=lambda: NOW)
    decision = guard.evaluate(proposal, state)
    assert not decision.approved
    assert any("market is closed" in r for r in decision.reasons)
