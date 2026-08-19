"""Scripted walkthrough of the Aegis pre-trade pipeline.

Runs offline: the option chain and the broker are both stubs, so no API keys
and no network are needed. Everything else -- the mandate, the verifier, the
risk guard, the approval tokens, the gateway -- is the real code.

    python demo.py
"""

from __future__ import annotations

import pathlib
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import yaml
from alpaca.trading.enums import OrderSide, PositionIntent

from aegis.core.approval import ApprovalToken
from aegis.core.gateway import ExecutionGateway, GovernanceError
from aegis.core.option_chain import Chain, Contract
from aegis.core.proposal import OrderLeg, PortfolioState, TradeProposal
from aegis.core.risk import RiskGuard, SessionState
from aegis.core.verifier import ObservationVerifier

NOW = datetime(2026, 8, 19, 15, 0, tzinfo=timezone.utc)  # 11:00 ET, mid-session
TODAY = NOW.date()
EXPIRY = date(2026, 9, 18)
SHORT, LONG = "SPY260918P00753000", "SPY260918P00748000"

MANDATE = yaml.safe_load(pathlib.Path("config/mandate.yaml").read_text())


# -- stubs: the two places that would otherwise touch the network -----------


def stub_chain(short_delta=-0.3014, short_oi=2804):
    """The market's version of the truth."""

    def fetch(underlying, **kwargs):
        return Chain(
            underlying=underlying,
            spot=767.35,
            expiration=EXPIRY,
            dte=30,
            contracts=(
                Contract(SHORT, "put", Decimal("753"), EXPIRY, 6.50, 6.79,
                         short_delta, 0.154, short_oi),
                Contract(LONG, "put", Decimal("748"), EXPIRY, 5.57, 5.58,
                         -0.2210, 0.159, 2241),
            ),
        )

    return fetch


class Broker:
    def __init__(self):
        self.orders = []

    def submit_order(self, order):
        self.orders.append(order)
        return type("Order", (), {"id": "ord-4417", "status": "accepted"})()


class Session:
    def state(self, now):
        return SessionState(
            is_open=True,
            opened_at=datetime(2026, 8, 19, 13, 30, tzinfo=timezone.utc),
        )


class Audit:
    def __init__(self):
        self.entries = []

    def record(self, event, payload):
        self.entries.append((event, payload))
        return f"hash-{len(self.entries)}"


class Portfolio:
    def fetch(self):
        return PortfolioState(
            equity=Decimal("100000"),
            buying_power=Decimal("400000"),
            open_positions=0,
            positions_by_symbol={},
            portfolio_delta=0.0,
            capital_at_risk_pct=0.0,
            fetched_at=NOW,
        )


# -- the proposal under test ------------------------------------------------

PROPOSAL = TradeProposal(
    proposal_id="prop-001",
    underlying="SPY",
    structure="vertical_credit_spread",
    legs=(
        OrderLeg(SHORT, OrderSide.SELL, PositionIntent.SELL_TO_OPEN, delta=Decimal("-0.30")),
        OrderLeg(LONG, OrderSide.BUY, PositionIntent.BUY_TO_OPEN, delta=Decimal("-0.22")),
    ),
    quantity=1,
    limit_price=Decimal("1.10"),
    max_loss_usd=Decimal("390"),
)


def run(label, proposal, *, chain=None, token_age=0):
    broker, audit = Broker(), Audit()
    gateway = ExecutionGateway(
        MANDATE,
        RiskGuard(MANDATE, session=Session(), clock=lambda: NOW),
        audit,
        broker,
        verifier=ObservationVerifier.from_mandate(
            MANDATE, chain or stub_chain(), today=lambda: TODAY
        ),
        clock=lambda: NOW,
    )
    token = ApprovalToken.issue(proposal, "demo-operator", now=NOW - timedelta(seconds=token_age))

    print(f"\n{label}")
    print("-" * len(label))
    try:
        result = gateway.submit(proposal, Portfolio(), token)
        print(f"  SUBMITTED   broker order {result.broker_order_id}, "
              f"client id {result.client_order_id}")
    except GovernanceError as exc:
        print(f"  REFUSED     {type(exc).__name__}")
        print(f"              {exc}")
    print(f"  orders sent to broker: {len(broker.orders)}")
    print(f"  audit chain: {[event for event, _ in audit.entries]}")


if __name__ == "__main__":
    print("Aegis pre-trade pipeline: mandate -> verifier -> risk guard -> token -> broker")

    run("1. Compliant proposal, claims corroborated by the market", PROPOSAL)

    run("2. Agent understates its short-leg delta (claims 0.10, market says 0.45)",
        replace(PROPOSAL,
                legs=(replace(PROPOSAL.legs[0], delta=Decimal("-0.10")), PROPOSAL.legs[1])),
        chain=stub_chain(short_delta=-0.45))

    run("3. Short leg is illiquid (open interest 12, mandate floor is 500)",
        PROPOSAL, chain=stub_chain(short_oi=12))

    run("4. Size breaches the risk limits (20 contracts, $7,800 max loss)",
        replace(PROPOSAL, quantity=20, max_loss_usd=Decimal("7800")))

    run("5. Human approval has gone stale (token issued 200s ago, TTL is 120s)",
        PROPOSAL, token_age=200)

    print("\nEvery refusal is written to the audit chain before the exception is raised.")
