"""Live portfolio state, read from Alpaca.

Lives here rather than in the runner because more than one surface needs it:
the CLI passes it to the gateway as a ``PortfolioSource``, and the web console
reads it to show the operator what the guard is measuring against.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from alpaca.data.requests import OptionSnapshotRequest
from alpaca.trading.enums import AssetClass

from aegis.core.option_chain import Clients
from aegis.core.proposal import PortfolioState, parse_occ_symbol


class LivePortfolio:
    """Fetches current portfolio state from Alpaca.

    This is the gateway's ``PortfolioSource``: it is called again inside
    ``submit()`` so the risk guard always evaluates a fresh snapshot, not the
    one the operator was looking at when they approved.
    """

    def __init__(self, clients: Clients) -> None:
        self._clients = clients

    def fetch(self) -> PortfolioState:
        account = self._clients.trading.get_account()
        positions = self._clients.trading.get_all_positions() or []
        equity = Decimal(str(account.equity))

        by_symbol: dict[str, int] = {}
        for position in positions:
            underlying = _underlying_of(position)
            by_symbol[underlying] = 1  # one logical position per symbol; see README

        capital_at_risk = Decimal(str(account.maintenance_margin or 0))
        return PortfolioState(
            equity=equity,
            buying_power=Decimal(str(account.buying_power)),
            open_positions=len(by_symbol),
            positions_by_symbol=by_symbol,
            portfolio_delta=float(self._portfolio_delta(positions)),
            capital_at_risk_pct=float(capital_at_risk / equity * 100) if equity > 0 else 0.0,
            fetched_at=datetime.now(timezone.utc),
        )

    def _portfolio_delta(self, positions) -> Decimal:
        """Share-equivalent delta across open option positions.

        Refuses to guess: an option position whose delta cannot be observed
        aborts the run rather than being silently counted as zero, which would
        understate portfolio delta and could let the guard pass a trade it
        should refuse.
        """
        option_symbols = [
            p.symbol for p in positions if p.asset_class == AssetClass.US_OPTION
        ]
        if not option_symbols:
            return Decimal(0)

        snapshots = self._clients.option.get_option_snapshot(
            OptionSnapshotRequest(symbol_or_symbols=option_symbols)
        )
        total = Decimal(0)
        for position in positions:
            if position.asset_class != AssetClass.US_OPTION:
                continue
            greeks = getattr(snapshots.get(position.symbol), "greeks", None)
            delta = getattr(greeks, "delta", None)
            if delta is None:
                raise RuntimeError(
                    f"no observed delta for open position {position.symbol}; "
                    "refusing to run with an understated portfolio delta"
                )
            total += Decimal(str(delta)) * Decimal(str(position.qty)) * 100
        return total


def _underlying_of(position) -> str:
    if position.asset_class == AssetClass.US_OPTION:
        try:
            return parse_occ_symbol(position.symbol).root
        except ValueError:
            return position.symbol
    return position.symbol




def _underlying_of(position) -> str:
    if position.asset_class == AssetClass.US_OPTION:
        try:
            return parse_occ_symbol(position.symbol).root
        except ValueError:
            return position.symbol
    return position.symbol
