"""The gateway's job is to refuse. These tests mostly prove it refuses.

No test here is allowed to reach a real broker: the trading client is a fake
that records what it was asked to do.
"""

from __future__ import annotations

import pathlib
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
import requests.exceptions
import yaml
from alpaca.trading.enums import OrderClass, OrderSide, PositionIntent

from aegis.core.approval import ApprovalToken
from aegis.core.gateway import (
    ApprovalExpired,
    VerificationFailed,
    ApprovalMismatch,
    ApprovalMissing,
    BrokerSubmissionError,
    ExecutionGateway,
    GovernanceError,
    MandateViolation,
    RiskRejected,
)
from aegis.core.proposal import OrderLeg, PortfolioState, RiskDecision, TradeProposal
from aegis.core.verifier import Discrepancy, VerificationResult

T0 = datetime(2026, 8, 19, 14, 0, tzinfo=timezone.utc)
MANDATE_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "mandate.yaml"


# --------------------------------------------------------------------------
# doubles
# --------------------------------------------------------------------------


class FakeOrder:
    def __init__(self, order_id: str = "ord-123", status: str = "accepted") -> None:
        self.id = order_id
        self.status = status


class FakeTradingClient:
    """Records submissions. Can be told to fail the first N attempts."""

    def __init__(self, fail_times: int = 0, error: Exception | None = None) -> None:
        self.orders: list = []
        self.fail_times = fail_times
        self.error = error or requests.exceptions.ConnectionError(
            "Temporary failure in name resolution"
        )

    def submit_order(self, order):
        self.orders.append(order)
        if len(self.orders) <= self.fail_times:
            raise self.error
        return FakeOrder()


class StubVerifier:
    """Corroborates everything, unless handed discrepancies or an exception."""

    def __init__(self, discrepancies=(), raises=None) -> None:
        self.discrepancies = tuple(discrepancies)
        self.raises = raises
        self.calls = 0

    def verify(self, proposal):
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return VerificationResult(
            verified=not self.discrepancies, discrepancies=self.discrepancies
        )


class FakeGuard:
    def __init__(self, decision: RiskDecision) -> None:
        self.decision = decision
        self.calls: list[tuple[TradeProposal, PortfolioState]] = []

    def evaluate(self, proposal, state) -> RiskDecision:
        self.calls.append((proposal, state))
        return self.decision


class FakeAudit:
    def __init__(self) -> None:
        self.entries: list[tuple[str, dict]] = []

    def record(self, event, payload):
        self.entries.append((event, dict(payload)))
        return f"hash-{len(self.entries)}"

    def events(self) -> list[str]:
        return [event for event, _ in self.entries]


class FakeSource:
    """A portfolio source that counts how often it was actually consulted."""

    def __init__(self, state: PortfolioState) -> None:
        self.state = state
        self.fetches = 0

    def fetch(self) -> PortfolioState:
        self.fetches += 1
        return self.state


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def mandate() -> dict:
    return yaml.safe_load(MANDATE_PATH.read_text())


@pytest.fixture
def proposal() -> TradeProposal:
    """A 753/748 put credit spread on SPY -- inside the real mandate."""
    return TradeProposal(
        proposal_id="prop-001",
        underlying="SPY",
        structure="vertical_credit_spread",
        legs=(
            OrderLeg("SPY260918P00753000", OrderSide.SELL, PositionIntent.SELL_TO_OPEN),
            OrderLeg("SPY260918P00748000", OrderSide.BUY, PositionIntent.BUY_TO_OPEN),
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


@pytest.fixture
def source(state) -> FakeSource:
    return FakeSource(state)


@pytest.fixture
def audit() -> FakeAudit:
    return FakeAudit()


@pytest.fixture
def client() -> FakeTradingClient:
    return FakeTradingClient()


def build_gateway(
    mandate, audit, client, *, approved=True, reasons=(), now=T0, verifier=None, **kw
):
    guard = FakeGuard(RiskDecision(approved=approved, reasons=tuple(reasons)))
    gateway = ExecutionGateway(
        mandate,
        guard,
        audit,
        client,
        verifier=verifier if verifier is not None else StubVerifier(),
        clock=lambda: now,
        sleeper=lambda _seconds: None,
        **kw,
    )
    return gateway, guard


# --------------------------------------------------------------------------
# required refusals
# --------------------------------------------------------------------------


def test_submit_without_token_raises(mandate, proposal, source, audit, client):
    gateway, _ = build_gateway(mandate, audit, client)

    with pytest.raises(ApprovalMissing):
        gateway.submit(proposal, source, None)

    assert client.orders == []


def test_submit_with_expired_token_raises(mandate, proposal, source, audit, client):
    token = ApprovalToken.issue(proposal, "lalit", now=T0)
    # 121s later: one second past the 120s TTL.
    gateway, _ = build_gateway(mandate, audit, client, now=T0 + timedelta(seconds=121))

    with pytest.raises(ApprovalExpired):
        gateway.submit(proposal, source, token)

    assert client.orders == []


def test_submit_with_token_for_a_different_proposal_raises(
    mandate, proposal, source, audit, client
):
    other = TradeProposal(
        proposal_id="prop-999",
        underlying="SPY",
        structure="vertical_credit_spread",
        legs=proposal.legs,
        quantity=1,
        limit_price=Decimal("1.10"),
        max_loss_usd=Decimal("390"),
    )
    token = ApprovalToken.issue(other, "lalit", now=T0)
    gateway, _ = build_gateway(mandate, audit, client)

    with pytest.raises(ApprovalMismatch):
        gateway.submit(proposal, source, token)

    assert client.orders == []


def test_submit_with_proposal_mutated_after_approval_raises(
    mandate, proposal, source, audit, client
):
    token = ApprovalToken.issue(proposal, "lalit", now=T0)

    # Same proposal_id, but the size was raised after the human signed off.
    mutated = TradeProposal(
        proposal_id=proposal.proposal_id,
        underlying=proposal.underlying,
        structure=proposal.structure,
        legs=proposal.legs,
        quantity=10,
        limit_price=proposal.limit_price,
        max_loss_usd=proposal.max_loss_usd,
    )
    gateway, _ = build_gateway(mandate, audit, client)

    with pytest.raises(ApprovalMismatch) as excinfo:
        gateway.submit(mutated, source, token)

    assert "changed after approval" in str(excinfo.value)
    assert client.orders == []


def test_submit_with_failing_risk_decision_raises_despite_valid_token(
    mandate, proposal, source, audit, client
):
    token = ApprovalToken.issue(proposal, "lalit", now=T0)
    gateway, guard = build_gateway(
        mandate, audit, client, approved=False, reasons=("max_loss_per_trade_usd exceeded",)
    )

    with pytest.raises(RiskRejected) as excinfo:
        gateway.submit(proposal, source, token)

    assert "max_loss_per_trade_usd exceeded" in str(excinfo.value)
    assert guard.calls, "the guard must be consulted, not bypassed"
    assert client.orders == []


def test_all_refusals_are_governance_errors(mandate, proposal, source, audit, client):
    """Every refusal path raises GovernanceError, never a quiet return."""
    gateway, _ = build_gateway(mandate, audit, client)
    for error_type in (ApprovalMissing, ApprovalExpired, ApprovalMismatch, RiskRejected):
        assert issubclass(error_type, GovernanceError)

    with pytest.raises(GovernanceError):
        gateway.submit(proposal, source, None)


# --------------------------------------------------------------------------
# happy path
# --------------------------------------------------------------------------


def test_happy_path_calls_trading_client_exactly_once(
    mandate, proposal, source, audit, client
):
    token = ApprovalToken.issue(proposal, "lalit", now=T0)
    gateway, _ = build_gateway(mandate, audit, client)

    result = gateway.submit(proposal, source, token)

    assert len(client.orders) == 1
    assert result.broker_order_id == "ord-123"
    assert result.proposal_id == "prop-001"
    assert result.operator == "lalit"
    assert result.proposal_hash == proposal.content_hash()


def test_happy_path_logs_submitted_with_broker_order_id(
    mandate, proposal, source, audit, client
):
    token = ApprovalToken.issue(proposal, "lalit", now=T0)
    gateway, _ = build_gateway(mandate, audit, client)

    gateway.submit(proposal, source, token)

    assert "SUBMITTED" in audit.events()
    entry = dict(audit.entries[-1][1])
    assert entry["broker_order_id"] == "ord-123"
    assert entry["proposal_hash"] == proposal.content_hash()
    assert entry["operator"] == "lalit"


def test_order_is_built_as_a_multi_leg_limit_order(mandate, proposal, source, audit, client):
    token = ApprovalToken.issue(proposal, "lalit", now=T0)
    gateway, _ = build_gateway(mandate, audit, client)

    gateway.submit(proposal, source, token)

    order = client.orders[0]
    assert order.order_class == OrderClass.MLEG
    assert order.qty == 1
    assert order.limit_price == 1.10
    assert [leg.symbol for leg in order.legs] == [
        "SPY260918P00753000",
        "SPY260918P00748000",
    ]
    assert [leg.side for leg in order.legs] == [OrderSide.SELL, OrderSide.BUY]
    # Deterministic id so a retry cannot double-fill.
    assert order.client_order_id.startswith("aegis-prop-001-")


# --------------------------------------------------------------------------
# the gateway trusts nothing it is handed
# --------------------------------------------------------------------------


def test_guard_is_re_run_against_a_freshly_fetched_state(
    mandate, proposal, source, audit, client
):
    token = ApprovalToken.issue(proposal, "lalit", now=T0)
    gateway, guard = build_gateway(mandate, audit, client)

    gateway.submit(proposal, source, token)

    assert source.fetches == 1, "state must be fetched at submit time"
    assert len(guard.calls) == 1
    _, evaluated_state = guard.calls[0]
    assert evaluated_state is source.state


def test_refusal_is_audited_before_raising(mandate, proposal, source, audit, client):
    gateway, _ = build_gateway(mandate, audit, client)

    with pytest.raises(GovernanceError):
        gateway.submit(proposal, source, None)

    assert audit.events() == ["REJECTED"]
    entry = audit.entries[0][1]
    assert entry["error"] == "ApprovalMissing"
    assert entry["proposal_id"] == "prop-001"


def test_symbol_outside_the_mandate_universe_is_refused(mandate, source, audit, client):
    off_mandate = TradeProposal(
        proposal_id="prop-002",
        underlying="GME",
        structure="vertical_credit_spread",
        legs=(OrderLeg("GME260918P00020000", OrderSide.SELL, PositionIntent.SELL_TO_OPEN),),
        quantity=1,
        limit_price=Decimal("0.50"),
        max_loss_usd=Decimal("100"),
    )
    token = ApprovalToken.issue(off_mandate, "lalit", now=T0)
    gateway, _ = build_gateway(mandate, audit, client)

    with pytest.raises(MandateViolation):
        gateway.submit(off_mandate, source, token)

    assert client.orders == []


def test_prohibited_structure_is_refused(mandate, source, audit, client):
    naked = TradeProposal(
        proposal_id="prop-003",
        underlying="SPY",
        structure="naked_short_put",
        legs=(OrderLeg("SPY260918P00753000", OrderSide.SELL, PositionIntent.SELL_TO_OPEN),),
        quantity=1,
        limit_price=Decimal("1.10"),
        max_loss_usd=Decimal("75000"),
    )
    token = ApprovalToken.issue(naked, "lalit", now=T0)
    gateway, _ = build_gateway(mandate, audit, client)

    with pytest.raises(MandateViolation):
        gateway.submit(naked, source, token)

    assert client.orders == []


def test_empty_mandate_fails_closed(proposal, source, audit, client):
    gateway, _ = build_gateway({}, audit, client)
    token = ApprovalToken.issue(proposal, "lalit", now=T0)

    with pytest.raises(MandateViolation):
        gateway.submit(proposal, source, token)


# --------------------------------------------------------------------------
# transient broker failures
# --------------------------------------------------------------------------


def test_transient_connect_error_is_retried_then_succeeds(
    mandate, proposal, source, audit
):
    client = FakeTradingClient(fail_times=2)  # succeeds on the third attempt
    token = ApprovalToken.issue(proposal, "lalit", now=T0)
    gateway, _ = build_gateway(mandate, audit, client)

    result = gateway.submit(proposal, source, token)

    assert len(client.orders) == 3
    assert result.broker_order_id == "ord-123"
    assert audit.events().count("BROKER_RETRY") == 2


def test_retries_reuse_one_client_order_id_so_a_retry_cannot_double_fill(
    mandate, proposal, source, audit
):
    client = FakeTradingClient(fail_times=2)
    token = ApprovalToken.issue(proposal, "lalit", now=T0)
    gateway, _ = build_gateway(mandate, audit, client)

    gateway.submit(proposal, source, token)

    ids = {order.client_order_id for order in client.orders}
    assert len(ids) == 1


def test_broker_unreachable_after_three_attempts_raises(mandate, proposal, source, audit):
    client = FakeTradingClient(fail_times=99)
    token = ApprovalToken.issue(proposal, "lalit", now=T0)
    gateway, _ = build_gateway(mandate, audit, client)

    with pytest.raises(BrokerSubmissionError):
        gateway.submit(proposal, source, token)

    assert len(client.orders) == 3
    assert "REJECTED" in audit.events()


def test_backoff_is_exponential(mandate, proposal, source, audit):
    client = FakeTradingClient(fail_times=2)
    token = ApprovalToken.issue(proposal, "lalit", now=T0)
    delays: list[float] = []
    guard = FakeGuard(RiskDecision(approved=True))
    gateway = ExecutionGateway(
        mandate,
        guard,
        audit,
        client,
        verifier=StubVerifier(),
        clock=lambda: T0,
        sleeper=delays.append,
        backoff_seconds=0.5,
    )

    gateway.submit(proposal, source, token)

    assert delays == [0.5, 1.0]


def test_a_non_transient_broker_error_is_not_retried(mandate, proposal, source, audit):
    client = FakeTradingClient(fail_times=99, error=ValueError("insufficient buying power"))
    token = ApprovalToken.issue(proposal, "lalit", now=T0)
    gateway, _ = build_gateway(mandate, audit, client)

    with pytest.raises(BrokerSubmissionError):
        gateway.submit(proposal, source, token)

    assert len(client.orders) == 1, "a broker refusal is a decision, not a blip"


# --------------------------------------------------------------------------
# observation verification
# --------------------------------------------------------------------------


def test_gateway_without_a_verifier_fails_closed(mandate, proposal, source, audit, client):
    gateway = ExecutionGateway(
        mandate,
        FakeGuard(RiskDecision(approved=True)),
        audit,
        client,
        verifier=None,
        clock=lambda: T0,
    )
    token = ApprovalToken.issue(proposal, "lalit", now=T0)

    with pytest.raises(VerificationFailed) as excinfo:
        gateway.submit(proposal, source, token)

    # Pin the deliberate refusal, not an incidental AttributeError on None.
    assert "no observation verifier configured" in str(excinfo.value)
    assert audit.entries[0][1]["reason"].startswith("no observation verifier")
    assert client.orders == []


def test_verification_discrepancy_refuses_and_is_audited(
    mandate, proposal, source, audit, client
):
    verifier = StubVerifier(
        discrepancies=(
            Discrepancy(
                "SPY260918P00753000", "delta", Decimal("-0.10"), Decimal("-0.45"),
                "claimed -0.10 but observed -0.45",
            ),
        )
    )
    gateway, _ = build_gateway(mandate, audit, client, verifier=verifier)
    token = ApprovalToken.issue(proposal, "lalit", now=T0)

    with pytest.raises(VerificationFailed):
        gateway.submit(proposal, source, token)

    assert client.orders == []
    assert audit.events() == ["REJECTED"]
    entry = audit.entries[0][1]
    assert entry["error"] == "VerificationFailed"
    assert entry["discrepancies"][0]["field"] == "delta"
    assert entry["discrepancies"][0]["observed"] == "-0.45"


def test_verifier_exception_is_a_governance_error_not_a_crash(
    mandate, proposal, source, audit, client
):
    verifier = StubVerifier(raises=RuntimeError("chain endpoint exploded"))
    gateway, _ = build_gateway(mandate, audit, client, verifier=verifier)
    token = ApprovalToken.issue(proposal, "lalit", now=T0)

    with pytest.raises(VerificationFailed) as excinfo:
        gateway.submit(proposal, source, token)

    assert "chain endpoint exploded" in str(excinfo.value)
    assert client.orders == []


def test_verification_runs_before_the_guard(mandate, proposal, source, audit, client):
    """A lying proposal is refused on its claims, not on their consequences."""
    verifier = StubVerifier(
        discrepancies=(
            Discrepancy("SPY260918P00753000", "open_interest", None, 12, "under floor"),
        )
    )
    gateway, guard = build_gateway(mandate, audit, client, verifier=verifier)
    token = ApprovalToken.issue(proposal, "lalit", now=T0)

    with pytest.raises(VerificationFailed):
        gateway.submit(proposal, source, token)

    assert guard.calls == [], "the guard should not run on unverified claims"
    assert source.fetches == 0
