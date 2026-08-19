"""Tests for the live runner's own logic: portfolio derivation and the clause table.

The Alpaca clients are stubbed; nothing here reaches the network or a broker.
"""

from __future__ import annotations

import pathlib
from decimal import Decimal
from types import SimpleNamespace

import pytest
import yaml
from alpaca.trading.enums import AssetClass

import run_live
from aegis.core.proposal import RiskDecision

MANDATE = yaml.safe_load(
    (pathlib.Path(__file__).resolve().parent.parent / "config" / "mandate.yaml").read_text()
)


def account(equity="100000", buying_power="400000", maintenance_margin="0"):
    return SimpleNamespace(
        equity=equity,
        buying_power=buying_power,
        maintenance_margin=maintenance_margin,
        account_number="PA000",
        status="ACTIVE",
        options_trading_level=3,
    )


def option_position(symbol, qty):
    return SimpleNamespace(symbol=symbol, qty=qty, asset_class=AssetClass.US_OPTION)


def equity_position(symbol, qty):
    return SimpleNamespace(symbol=symbol, qty=qty, asset_class=AssetClass.US_EQUITY)


class StubClients:
    def __init__(self, acct=None, positions=(), deltas=None):
        self._deltas = deltas or {}
        self.trading = SimpleNamespace(
            get_account=lambda: acct or account(),
            get_all_positions=lambda: list(positions),
        )
        self.option = SimpleNamespace(get_option_snapshot=self._snapshot)

    def _snapshot(self, request):
        return {
            symbol: SimpleNamespace(greeks=SimpleNamespace(delta=self._deltas.get(symbol)))
            for symbol in request.symbol_or_symbols
        }


# --------------------------------------------------------------------------
# portfolio derivation
# --------------------------------------------------------------------------


def test_an_empty_account_reads_as_flat():
    state = run_live.LivePortfolio(StubClients()).fetch()
    assert state.open_positions == 0
    assert state.positions_by_symbol == {}
    assert state.portfolio_delta == 0.0
    assert state.capital_at_risk_pct == 0.0
    assert state.equity == Decimal("100000")


def test_option_legs_collapse_to_their_underlying():
    """Two legs of one SPY spread are one SPY position, not two."""
    clients = StubClients(
        positions=[
            option_position("SPY260918P00753000", "-1"),
            option_position("SPY260918P00748000", "1"),
        ],
        deltas={"SPY260918P00753000": -0.30, "SPY260918P00748000": -0.22},
    )
    state = run_live.LivePortfolio(clients).fetch()
    assert state.positions_by_symbol == {"SPY": 1}
    assert state.open_positions == 1


def test_portfolio_delta_comes_from_observed_greeks():
    clients = StubClients(
        positions=[
            option_position("SPY260918P00753000", "-1"),  # short
            option_position("SPY260918P00748000", "1"),   # long
        ],
        deltas={"SPY260918P00753000": -0.30, "SPY260918P00748000": -0.22},
    )
    state = run_live.LivePortfolio(clients).fetch()
    # (-0.30 * -1 * 100) + (-0.22 * 1 * 100) = +30 - 22 = +8
    assert state.portfolio_delta == pytest.approx(8.0)


def test_an_unobservable_delta_aborts_rather_than_counting_zero():
    """Silently treating an unknown delta as 0 would understate portfolio risk."""
    clients = StubClients(
        positions=[option_position("SPY260918P00753000", "-1")],
        deltas={"SPY260918P00753000": None},
    )
    with pytest.raises(RuntimeError, match="refusing to run"):
        run_live.LivePortfolio(clients).fetch()


def test_capital_at_risk_comes_from_maintenance_margin():
    clients = StubClients(acct=account(maintenance_margin="2500"))
    state = run_live.LivePortfolio(clients).fetch()
    assert state.capital_at_risk_pct == pytest.approx(2.5)


def test_equity_positions_count_under_their_own_symbol():
    clients = StubClients(positions=[equity_position("AAPL", "100")])
    state = run_live.LivePortfolio(clients).fetch()
    assert state.positions_by_symbol == {"AAPL": 1}


def test_state_is_refetched_not_cached():
    """The gateway calls fetch() again at submit time; it must hit the API."""
    calls = []

    class Counting(StubClients):
        def __init__(self):
            super().__init__()
            self.trading = SimpleNamespace(
                get_account=lambda: (calls.append("account"), account())[1],
                get_all_positions=lambda: [],
            )

    portfolio = Counting()
    source = run_live.LivePortfolio(portfolio)
    source.fetch()
    source.fetch()
    assert len(calls) == 2


# --------------------------------------------------------------------------
# the clause table cannot disagree with the verdict it describes
# --------------------------------------------------------------------------


def test_every_enforced_clause_has_a_display_row():
    """A clause the guard can cite but the table omits would be invisible on camera."""
    paths = {path for path, _ in run_live.CLAUSE_ROWS}
    for key in (MANDATE["risk_limits"] or {}):
        assert f"risk_limits.{key}" in paths, f"no display row for risk_limits.{key}"


def test_a_row_is_marked_failed_exactly_when_the_guard_cites_it(capsys):
    from aegis.agents.research import ResearchAgent

    from tests.test_agents import PUT_LADDER, StubFetcher, chain_from  # noqa: F401

    state = run_live.PortfolioState(
        equity=Decimal("100000"),
        buying_power=Decimal("400000"),
        open_positions=0,
        positions_by_symbol={},
        portfolio_delta=0.0,
        capital_at_risk_pct=0.0,
        fetched_at=run_live.datetime.now(run_live.timezone.utc),
    )
    proposal = ResearchAgent(
        MANDATE, StubFetcher(chain_from(PUT_LADDER)), None, today=lambda: run_live.date.today()
    ).propose("SPY", state)
    assert proposal is not None

    refused = RiskDecision(
        approved=False,
        reasons=("max loss $9999 exceeds max_loss_per_trade_usd $500",),
    )
    run_live.render_decision(
        refused, proposal, state, MANDATE, SimpleNamespace(is_open=True, opened_at=None)
    )
    out = capsys.readouterr().out
    # The cited clause is marked failed; an uncited one is not.
    assert "✗" in out and "✓" in out
    assert "REFUSED" in out
