"""The guard is arithmetic, so these tests are mostly arithmetic too.

Each limit in config/mandate.yaml gets a test that trips it, plus tests for
the cases where the guard must refuse because it *cannot* check something.
"""

from __future__ import annotations

import pathlib
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
import yaml
from alpaca.trading.enums import OrderSide, PositionIntent

from aegis.core.proposal import OrderLeg, PortfolioState, TradeProposal
from aegis.core.risk import RiskGuard, SessionState, derive_max_loss, net_delta

MANDATE_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "mandate.yaml"

# 2026-08-19 14:00 UTC == 10:00 ET: open, well clear of both the opening
# settle window and the 15:30 cutoff.
T0 = datetime(2026, 8, 19, 14, 0, tzinfo=timezone.utc)
MARKET_OPEN = datetime(2026, 8, 19, 13, 30, tzinfo=timezone.utc)  # 09:30 ET


class FakeSession:
    def __init__(self, is_open=True, opened_at=MARKET_OPEN):
        self._state = SessionState(is_open=is_open, opened_at=opened_at)

    def state(self, now):
        return self._state


@pytest.fixture
def mandate() -> dict:
    return yaml.safe_load(MANDATE_PATH.read_text())


@pytest.fixture
def proposal() -> TradeProposal:
    """SPY 753/748 put credit spread, 30 DTE: comfortably inside every limit."""
    return TradeProposal(
        proposal_id="prop-001",
        underlying="SPY",
        structure="vertical_credit_spread",
        legs=(
            OrderLeg(
                "SPY260918P00753000",
                OrderSide.SELL,
                PositionIntent.SELL_TO_OPEN,
                delta=Decimal("-0.30"),
            ),
            OrderLeg(
                "SPY260918P00748000",
                OrderSide.BUY,
                PositionIntent.BUY_TO_OPEN,
                delta=Decimal("-0.22"),
            ),
        ),
        quantity=1,
        limit_price=Decimal("1.10"),
        max_loss_usd=Decimal("390"),
    )


@pytest.fixture
def state() -> PortfolioState:
    return PortfolioState(
        equity=Decimal("100000"),
        buying_power=Decimal("400000"),
        open_positions=0,
        positions_by_symbol={},
        portfolio_delta=0.0,
        capital_at_risk_pct=0.0,
        fetched_at=T0,
    )


def build_guard(mandate, *, now=T0, session=None) -> RiskGuard:
    return RiskGuard(
        mandate,
        session=session if session is not None else FakeSession(),
        clock=lambda: now,
    )


def why(decision) -> str:
    return " | ".join(decision.reasons)


# --------------------------------------------------------------------------
# baseline
# --------------------------------------------------------------------------


def test_a_compliant_proposal_is_approved(mandate, proposal, state):
    decision = build_guard(mandate).evaluate(proposal, state)
    assert decision.approved, why(decision)
    assert decision.reasons == ()


def test_evaluation_is_deterministic(mandate, proposal, state):
    guard = build_guard(mandate)
    assert guard.evaluate(proposal, state) == guard.evaluate(proposal, state)


# --------------------------------------------------------------------------
# risk_limits
# --------------------------------------------------------------------------


def test_max_loss_per_trade_usd(mandate, proposal, state):
    # 20-point wide spread: 2000 - 110 = 1890 max loss, over the 500 cap.
    wide = replace(
        proposal,
        legs=(proposal.legs[0], replace(proposal.legs[1], symbol="SPY260918P00733000")),
        max_loss_usd=Decimal("1890"),
    )
    decision = build_guard(mandate).evaluate(wide, state)
    assert not decision.approved
    assert "max_loss_per_trade_usd" in why(decision)


def test_max_loss_per_trade_pct_of_equity(mandate, proposal, state):
    poor = replace(state, equity=Decimal("20000"))  # 390 is 1.95% of equity
    decision = build_guard(mandate).evaluate(proposal, poor)
    assert not decision.approved
    assert "max_loss_per_trade_pct_of_equity" in why(decision)


def test_max_open_positions(mandate, proposal, state):
    full = replace(state, open_positions=5)
    decision = build_guard(mandate).evaluate(proposal, full)
    assert not decision.approved
    assert "max_open_positions" in why(decision)


def test_max_positions_per_symbol(mandate, proposal, state):
    held = replace(state, positions_by_symbol={"SPY": 1})
    decision = build_guard(mandate).evaluate(proposal, held)
    assert not decision.approved
    assert "max_positions_per_symbol" in why(decision)


def test_max_portfolio_delta_abs(mandate, proposal, state):
    loaded = replace(state, portfolio_delta=98.0)  # +8 from this trade -> 106
    decision = build_guard(mandate).evaluate(proposal, loaded)
    assert not decision.approved
    assert "max_portfolio_delta_abs" in why(decision)


def test_max_total_capital_at_risk_pct(mandate, proposal, state):
    committed = replace(state, capital_at_risk_pct=4.9)  # +0.39 -> 5.29%
    decision = build_guard(mandate).evaluate(proposal, committed)
    assert not decision.approved
    assert "max_total_capital_at_risk_pct" in why(decision)


def test_min_buying_power_buffer_usd(mandate, proposal, state):
    thin = replace(state, buying_power=Decimal("25100"))  # 25100 - 390 < 25000
    decision = build_guard(mandate).evaluate(proposal, thin)
    assert not decision.approved
    assert "min_buying_power_buffer_usd" in why(decision)


# --------------------------------------------------------------------------
# strategy limits
# --------------------------------------------------------------------------


def test_short_leg_delta_ceiling(mandate, proposal, state):
    hot = replace(
        proposal,
        legs=(replace(proposal.legs[0], delta=Decimal("-0.42")), proposal.legs[1]),
    )
    decision = build_guard(mandate).evaluate(hot, state)
    assert not decision.approved
    assert "short_leg_abs_delta_max" in why(decision)


def test_short_leg_exactly_at_the_delta_limit_is_allowed(mandate, proposal, state):
    at_limit = replace(
        proposal,
        legs=(replace(proposal.legs[0], delta=Decimal("-0.30")), proposal.legs[1]),
    )
    assert build_guard(mandate).evaluate(at_limit, state).approved


def test_long_leg_delta_is_not_capped(mandate, proposal, state):
    """Only the short leg carries assignment risk, so only it is capped."""
    deep_long = replace(
        proposal,
        legs=(proposal.legs[0], replace(proposal.legs[1], delta=Decimal("-0.85"))),
    )
    decision = build_guard(mandate).evaluate(deep_long, state)
    assert "short_leg_abs_delta_max" not in why(decision)


def test_min_credit_to_max_loss_ratio(mandate, proposal, state):
    # 0.50 credit against a 450 max loss -> 0.111, under the 0.20 floor.
    thin = replace(proposal, limit_price=Decimal("0.50"), max_loss_usd=Decimal("450"))
    decision = build_guard(mandate).evaluate(thin, state)
    assert not decision.approved
    assert "min_credit_to_max_loss_ratio" in why(decision)


def test_expiry_too_near(mandate, proposal, state):
    decision = build_guard(mandate, now=T0 + timedelta(days=27)).evaluate(proposal, state)
    assert not decision.approved
    assert "expiry_window_days.min" in why(decision)


def test_expiry_too_far(mandate, proposal, state):
    decision = build_guard(mandate, now=T0 - timedelta(days=20)).evaluate(proposal, state)
    assert not decision.approved
    assert "expiry_window_days.max" in why(decision)


def test_legs_spanning_multiple_expiries_are_refused(mandate, proposal, state):
    calendar = replace(
        proposal,
        legs=(proposal.legs[0], replace(proposal.legs[1], symbol="SPY260925P00748000")),
    )
    decision = build_guard(mandate).evaluate(calendar, state)
    assert not decision.approved
    assert "multiple expiries" in why(decision)


# --------------------------------------------------------------------------
# derive, do not trust
# --------------------------------------------------------------------------


def test_derived_max_loss_matches_the_strikes(proposal):
    assert derive_max_loss(proposal) == Decimal("390.00")


def test_understated_max_loss_is_caught(mandate, proposal, state):
    """A proposal claiming less risk than its strikes imply is refused."""
    liar = replace(proposal, max_loss_usd=Decimal("50"))
    decision = build_guard(mandate).evaluate(liar, state)
    assert not decision.approved
    assert "disagrees with" in why(decision)


def test_the_conservative_figure_is_used_for_limits(mandate, proposal, state):
    """Understating max loss must not buy a pass on the dollar limits."""
    wide = replace(
        proposal,
        legs=(proposal.legs[0], replace(proposal.legs[1], symbol="SPY260918P00733000")),
        max_loss_usd=Decimal("10"),  # real loss is 1890
    )
    decision = build_guard(mandate).evaluate(wide, state)
    assert not decision.approved
    assert "max_loss_per_trade_usd" in why(decision)


def test_underivable_structure_is_refused(mandate, proposal, state):
    exotic = replace(proposal, structure="calendar_spread")
    decision = build_guard(mandate).evaluate(exotic, state)
    assert not decision.approved
    assert "cannot derive max loss" in why(decision)


def test_missing_delta_is_refused(mandate, proposal, state):
    blind = replace(
        proposal, legs=(replace(proposal.legs[0], delta=None), proposal.legs[1])
    )
    decision = build_guard(mandate).evaluate(blind, state)
    assert not decision.approved
    assert "no delta" in why(decision)


def test_net_delta_signs_short_legs_negatively(proposal):
    # Short the 0.30 put, long the 0.22 put -> net +8 share-equivalents.
    assert net_delta(proposal) == Decimal("8.00")


def test_iron_condor_max_loss_uses_the_wider_wing():
    condor = TradeProposal(
        proposal_id="prop-ic",
        underlying="SPY",
        structure="iron_condor",
        legs=(
            OrderLeg("SPY260918P00750000", OrderSide.SELL, PositionIntent.SELL_TO_OPEN, delta=Decimal("-0.28")),
            OrderLeg("SPY260918P00745000", OrderSide.BUY, PositionIntent.BUY_TO_OPEN, delta=Decimal("-0.20")),
            OrderLeg("SPY260918C00790000", OrderSide.SELL, PositionIntent.SELL_TO_OPEN, delta=Decimal("0.21")),
            OrderLeg("SPY260918C00800000", OrderSide.BUY, PositionIntent.BUY_TO_OPEN, delta=Decimal("0.10")),
        ),
        quantity=1,
        limit_price=Decimal("2.00"),
        max_loss_usd=Decimal("800"),
    )
    # Wider wing is the 10-point call side: 1000 - 200 credit = 800.
    assert derive_max_loss(condor) == Decimal("800.00")


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------


def test_no_session_source_is_refused(mandate, proposal, state):
    guard = RiskGuard(mandate, session=None, clock=lambda: T0)
    decision = guard.evaluate(proposal, state)
    assert not decision.approved
    assert "no market session source" in why(decision)


def test_closed_market_is_refused(mandate, proposal, state):
    decision = build_guard(mandate, session=FakeSession(is_open=False)).evaluate(
        proposal, state
    )
    assert not decision.approved
    assert "market is closed" in why(decision)


def test_within_the_opening_settle_window_is_refused(mandate, proposal, state):
    just_open = MARKET_OPEN + timedelta(minutes=5)
    decision = build_guard(mandate, now=just_open).evaluate(proposal, state)
    assert not decision.approved
    assert "minutes of the open" in why(decision)


def test_before_the_open_does_not_claim_the_settle_window(mandate, proposal, state):
    """Pre-market is refused for being closed, not for being just after the open."""
    pre_open = MARKET_OPEN - timedelta(hours=4)
    decision = build_guard(
        mandate, now=pre_open, session=FakeSession(is_open=False)
    ).evaluate(proposal, state)

    assert not decision.approved
    assert any("market is closed" in r for r in decision.reasons)
    assert not any("minutes of the open" in r for r in decision.reasons)


def test_after_the_afternoon_cutoff_is_refused(mandate, proposal, state):
    late = datetime(2026, 8, 19, 19, 45, tzinfo=timezone.utc)  # 15:45 ET
    decision = build_guard(mandate, now=late).evaluate(proposal, state)
    assert not decision.approved
    assert "no_new_positions_after" in why(decision)


# --------------------------------------------------------------------------
# fail closed
# --------------------------------------------------------------------------


def test_empty_mandate_refuses_everything(proposal, state):
    decision = RiskGuard({}, session=FakeSession(), clock=lambda: T0).evaluate(
        proposal, state
    )
    assert not decision.approved
    assert "no risk_limits" in why(decision)


def test_all_violations_are_reported_not_just_the_first(mandate, proposal, state):
    bad_state = replace(
        state,
        open_positions=5,
        positions_by_symbol={"SPY": 1},
        buying_power=Decimal("100"),
    )
    decision = build_guard(mandate).evaluate(proposal, bad_state)
    assert not decision.approved
    assert len(decision.reasons) >= 3, why(decision)


# --------------------------------------------------------------------------
# the gateway actually accepts this guard
# --------------------------------------------------------------------------


def test_guard_satisfies_the_gateway_protocol(mandate, proposal, state):
    from aegis.core.approval import ApprovalToken
    from aegis.core.gateway import ExecutionGateway, RiskRejected

    class Source:
        def fetch(self):
            return state

    class Audit:
        def __init__(self):
            self.entries = []

        def record(self, event, payload):
            self.entries.append((event, payload))

    class Client:
        def __init__(self):
            self.orders = []

        def submit_order(self, order):
            self.orders.append(order)
            return type("O", (), {"id": "ord-9", "status": "accepted"})()

    class Verifier:
        def verify(self, proposal):
            from aegis.core.verifier import VerificationResult

            return VerificationResult(verified=True)

    client = Client()
    gateway = ExecutionGateway(
        mandate,
        build_guard(mandate),
        Audit(),
        client,
        verifier=Verifier(),
        clock=lambda: T0,
    )
    token = ApprovalToken.issue(proposal, "lalit", now=T0)

    result = gateway.submit(proposal, Source(), token)
    assert result.broker_order_id == "ord-9"
    assert len(client.orders) == 1

    # And the same wiring refuses when a limit is breached.
    over_limit = replace(proposal, quantity=20, max_loss_usd=Decimal("7800"))
    with pytest.raises(RiskRejected):
        gateway.submit(over_limit, Source(), ApprovalToken.issue(over_limit, "lalit", now=T0))
