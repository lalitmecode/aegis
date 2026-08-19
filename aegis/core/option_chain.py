"""Option chain retrieval, joined across the two Alpaca endpoints that carry it.

Neither endpoint is sufficient on its own:

* market data ``/options/snapshots`` carries quotes, implied volatility and
  greeks, but no open interest;
* trading ``/options/contracts`` carries open interest, but no greeks.

:func:`fetch_chain` issues one of each and joins them on the OCC symbol, so a
caller sees delta and open interest side by side -- the two fields
``config/mandate.yaml`` screens on (``delta_limits.short_leg_abs_delta_max``
and the 500-contract open interest floor under ``universe.prohibited``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import OptionChainRequest, StockLatestTradeRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetStatus, ContractType
from alpaca.trading.requests import GetOptionContractsRequest

# Defaults track config/mandate.yaml: strategy.expiry_window_days.
MANDATE_EXPIRY_WINDOW = (7, 45)
DEFAULT_TARGET_DTE = 30
DEFAULT_MONEYNESS = 0.05

_CONTRACTS_PAGE_LIMIT = 10_000


@dataclass(frozen=True)
class Clients:
    """The three Alpaca clients a chain pull needs."""

    trading: TradingClient
    stock: StockHistoricalDataClient
    option: OptionHistoricalDataClient

    @classmethod
    def from_env(cls, env_file: str | os.PathLike[str] | None = ".env") -> "Clients":
        """Build clients from ALPACA_* environment variables.

        Reads ``env_file`` first if it exists, so this works both from a shell
        with the variables exported and from a bare process in the repo root.
        """
        if env_file and os.path.exists(env_file):
            from dotenv import load_dotenv

            load_dotenv(env_file)

        key = os.environ.get("ALPACA_API_KEY")
        secret = os.environ.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set "
                f"(looked in the environment and {env_file!r})"
            )
        paper = os.environ.get("ALPACA_PAPER_TRADE", "true").strip().lower() != "false"

        return cls(
            trading=TradingClient(key, secret, paper=paper),
            stock=StockHistoricalDataClient(key, secret),
            option=OptionHistoricalDataClient(key, secret),
        )


@dataclass(frozen=True)
class Contract:
    """One option contract: terms, quote, greeks and open interest."""

    symbol: str
    type: str  # "call" or "put"
    strike: float
    expiration: date
    bid: float | None = None
    ask: float | None = None
    delta: float | None = None
    implied_volatility: float | None = None
    open_interest: int | None = None
    quote_time: datetime | None = None

    @property
    def mid(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid


@dataclass(frozen=True)
class Chain:
    """A single expiry's contracts within a strike band around spot."""

    underlying: str
    spot: float
    expiration: date
    dte: int
    contracts: tuple[Contract, ...]
    feed: str | None = None
    asof: datetime | None = None
    open_interest_asof: date | None = None

    def calls(self) -> list[Contract]:
        return [c for c in self.contracts if c.type == "call"]

    def puts(self) -> list[Contract]:
        return [c for c in self.contracts if c.type == "put"]

    def strikes(self) -> list[float]:
        return sorted({c.strike for c in self.contracts})

    def by_strike(self) -> dict[float, dict[str, Contract]]:
        """``{strike: {"call": Contract, "put": Contract}}``, ascending."""
        out: dict[float, dict[str, Contract]] = {}
        for c in self.contracts:
            out.setdefault(c.strike, {})[c.type] = c
        return {k: out[k] for k in sorted(out)}

    def nearest_delta(self, target: float, type: str) -> Contract | None:
        """Contract of ``type`` whose |delta| is closest to |target|.

        The usual way to pick a short leg against the mandate's
        ``short_leg_abs_delta_max``.
        """
        candidates = [c for c in self.contracts if c.type == type and c.delta is not None]
        if not candidates:
            return None
        return min(candidates, key=lambda c: abs(abs(c.delta) - abs(target)))

    def liquid(self, min_open_interest: int = 500) -> list[Contract]:
        """Contracts clearing the mandate's open interest floor."""
        return [
            c
            for c in self.contracts
            if c.open_interest is not None and c.open_interest >= min_open_interest
        ]

    def format_table(self) -> str:
        """Render as a paired call/put chain, one row per strike."""

        def num(v: float | None, places: int = 2) -> str:
            return "-" if v is None else f"{v:.{places}f}"

        def pct(v: float | None) -> str:
            return "-" if v is None else f"{v * 100:.1f}"

        def oi(v: int | None) -> str:
            return "-" if v is None else f"{v:,}"

        header = (
            f"{'CALL bid':>9} {'ask':>7} {'delta':>6} {'IV%':>6} {'OI':>8}"
            f"  |{'STRIKE':^8}|  "
            f"{'PUT bid':>8} {'ask':>7} {'delta':>6} {'IV%':>6} {'OI':>8}"
        )
        lines = [
            f"{self.underlying} {self.expiration} ({self.dte}d)  "
            f"spot {self.spot:.2f}  feed={self.feed or 'auto'}",
            header,
            "-" * len(header),
        ]
        for strike, side in self.by_strike().items():
            c, p = side.get("call"), side.get("put")
            marker = " <" if abs(strike - self.spot) < 0.5 else ""
            lines.append(
                f"{num(c and c.bid):>9} {num(c and c.ask):>7} "
                f"{num(c and c.delta, 3):>6} {pct(c and c.implied_volatility):>6} "
                f"{oi(c and c.open_interest):>8}"
                f"  |{strike:^8.0f}|  "
                f"{num(p and p.bid):>8} {num(p and p.ask):>7} "
                f"{num(p and p.delta, 3):>6} {pct(p and p.implied_volatility):>6} "
                f"{oi(p and p.open_interest):>8}{marker}"
            )
        return "\n".join(lines)


def fetch_chain(
    underlying: str,
    *,
    target_dte: int = DEFAULT_TARGET_DTE,
    moneyness: float = DEFAULT_MONEYNESS,
    expiry_window: tuple[int, int] = MANDATE_EXPIRY_WINDOW,
    contract_type: str | None = None,
    feed: str | None = None,
    clients: Clients | None = None,
    today: date | None = None,
) -> Chain:
    """Fetch the expiry nearest ``target_dte`` and the strikes around spot.

    Args:
        underlying: Underlying symbol, e.g. ``"SPY"``.
        target_dte: Preferred days to expiry; the closest listed expiry within
            ``expiry_window`` wins, ties breaking to the nearer date.
        moneyness: Half-width of the strike band as a fraction of spot, so
            ``0.05`` means strikes within +/-5%.
        expiry_window: ``(min_dte, max_dte)`` bounding which expiries are
            eligible. Defaults to the mandate's 7-45 day window.
        contract_type: ``"call"``, ``"put"``, or None for both.
        feed: ``"opra"`` or ``"indicative"``. None lets Alpaca choose, which
            is ``indicative`` unless the account has an OPRA subscription.
        clients: Reuse existing clients; built from the environment if None.
        today: Override the reference date, for testing.

    Raises:
        LookupError: No listed expiry falls inside ``expiry_window``.
        ValueError: ``moneyness`` or ``expiry_window`` is malformed.
    """
    if moneyness <= 0:
        raise ValueError(f"moneyness must be positive, got {moneyness}")
    min_dte, max_dte = expiry_window
    if min_dte > max_dte:
        raise ValueError(f"expiry_window is inverted: {expiry_window}")

    clients = clients or Clients.from_env()
    today = today or date.today()
    ctype = ContractType(contract_type) if contract_type else None

    spot = _latest_price(clients.stock, underlying)
    low = round(spot * (1 - moneyness), 2)
    high = round(spot * (1 + moneyness), 2)

    # One contracts call covers the whole eligible window; we pick the expiry
    # from it and reuse the same rows for open interest.
    contracts = _list_contracts(
        clients.trading,
        underlying=underlying,
        start=today + timedelta(days=min_dte),
        end=today + timedelta(days=max_dte),
        low=low,
        high=high,
        ctype=ctype,
    )
    if not contracts:
        raise LookupError(
            f"no {underlying} contracts listed between {min_dte} and {max_dte} days out "
            f"with strikes in {low}-{high}"
        )

    expiration = min(
        {c.expiration_date for c in contracts},
        key=lambda e: (abs((e - today).days - target_dte), e),
    )
    contracts = [c for c in contracts if c.expiration_date == expiration]

    snapshots = clients.option.get_option_chain(
        OptionChainRequest(
            underlying_symbol=underlying,
            expiration_date=expiration,
            strike_price_gte=low,
            strike_price_lte=high,
            type=ctype,
            feed=feed,
        )
    )

    rows = tuple(
        _join(contract, snapshots.get(contract.symbol))
        for contract in sorted(contracts, key=lambda c: (c.strike_price, c.type.value))
    )
    quote_times = [r.quote_time for r in rows if r.quote_time]
    oi_dates = {c.open_interest_date for c in contracts if c.open_interest_date}

    return Chain(
        underlying=underlying,
        spot=spot,
        expiration=expiration,
        dte=(expiration - today).days,
        contracts=rows,
        feed=feed,
        asof=max(quote_times) if quote_times else None,
        open_interest_asof=max(oi_dates) if oi_dates else None,
    )


def _latest_price(client: StockHistoricalDataClient, symbol: str) -> float:
    trade = client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))
    return float(trade[symbol].price)


def _list_contracts(
    client: TradingClient,
    *,
    underlying: str,
    start: date,
    end: date,
    low: float,
    high: float,
    ctype: ContractType | None,
):
    """Page through /v2/options/contracts, which does not auto-paginate."""
    out, page_token = [], None
    while True:
        response = client.get_option_contracts(
            GetOptionContractsRequest(
                underlying_symbols=[underlying],
                status=AssetStatus.ACTIVE,
                expiration_date_gte=start,
                expiration_date_lte=end,
                strike_price_gte=str(low),
                strike_price_lte=str(high),
                type=ctype,
                limit=_CONTRACTS_PAGE_LIMIT,
                page_token=page_token,
            )
        )
        out.extend(response.option_contracts or [])
        page_token = response.next_page_token
        if not page_token:
            return out


def _join(contract, snapshot) -> Contract:
    """Merge a contract record with its market data snapshot, if any."""
    quote = getattr(snapshot, "latest_quote", None)
    greeks = getattr(snapshot, "greeks", None)
    return Contract(
        symbol=contract.symbol,
        type=contract.type.value,
        strike=float(contract.strike_price),
        expiration=contract.expiration_date,
        bid=getattr(quote, "bid_price", None),
        ask=getattr(quote, "ask_price", None),
        delta=getattr(greeks, "delta", None),
        implied_volatility=getattr(snapshot, "implied_volatility", None),
        open_interest=int(contract.open_interest) if contract.open_interest else None,
        quote_time=getattr(quote, "timestamp", None),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("underlying")
    parser.add_argument("--dte", type=int, default=DEFAULT_TARGET_DTE)
    parser.add_argument("--moneyness", type=float, default=DEFAULT_MONEYNESS)
    parser.add_argument("--type", choices=("call", "put"), default=None)
    parser.add_argument("--feed", choices=("opra", "indicative"), default=None)
    args = parser.parse_args()

    chain = fetch_chain(
        args.underlying,
        target_dte=args.dte,
        moneyness=args.moneyness,
        contract_type=args.type,
        feed=args.feed,
    )
    print(chain.format_table())
    print(
        f"\n{len(chain.strikes())} strikes, {len(chain.contracts)} contracts"
        f"  quotes asof {chain.asof}  open interest asof {chain.open_interest_asof}"
    )
