"""Value objects shared by the critic, the risk guard and the execution gateway.

The important one is :class:`TradeProposal`, which is *content-addressable*:
:meth:`TradeProposal.content_hash` is a stable digest of every field that
affects execution or risk. Human approval is bound to that digest, so a
proposal edited after approval no longer matches the token that approved it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Mapping

from alpaca.trading.enums import OrderSide, PositionIntent, TimeInForce


def _enum_value(value: Any) -> Any:
    """Normalise an enum-or-string to its wire value."""
    return getattr(value, "value", value)


def _number(value: Any) -> str:
    """Normalise any numeric to a canonical decimal string.

    ``1.25``, ``Decimal("1.25")`` and ``"1.25"`` must all hash identically,
    otherwise a token could be voided by a lossless round-trip.
    """
    return format(Decimal(str(value)).normalize(), "f")


@dataclass(frozen=True, slots=True)
class OrderLeg:
    """One leg of a multi-leg options order."""

    symbol: str  # OCC contract symbol, e.g. SPY260918P00753000
    side: OrderSide
    position_intent: PositionIntent
    ratio_qty: int = 1
    #: Delta observed when the proposal was built. Hashed like every other
    #: field: the risk guard reads it, so it must not be changeable after
    #: approval without voiding the token.
    delta: Decimal | None = None

    def canonical(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": _enum_value(self.side),
            "position_intent": _enum_value(self.position_intent),
            "ratio_qty": int(self.ratio_qty),
            "delta": None if self.delta is None else _number(self.delta),
        }

    @property
    def is_short(self) -> bool:
        return _enum_value(self.side) == _enum_value(OrderSide.SELL)


@dataclass(frozen=True, slots=True)
class TradeProposal:
    """A defined-risk structure the agent wants to open.

    Every field here is hashed. There is deliberately no free-form metadata
    field: anything excluded from the hash would be a channel for changing a
    proposal without voiding its approval.
    """

    proposal_id: str
    underlying: str
    structure: str  # must appear in mandate.strategy.permitted_structures
    legs: tuple[OrderLeg, ...]
    quantity: int
    limit_price: Decimal
    max_loss_usd: Decimal
    time_in_force: TimeInForce = TimeInForce.DAY
    #: The research agent's written rationale. Hashed with everything else: a
    #: human approves a trade *and* the reasoning given for it, so rewriting
    #: the thesis after approval voids the token.
    thesis: str | None = None

    def canonical_payload(self) -> dict[str, Any]:
        """Deterministic dict of the proposal's executable content.

        Leg order is preserved rather than sorted -- reordering legs changes
        the order, so it must change the hash.
        """
        return {
            "proposal_id": self.proposal_id,
            "underlying": self.underlying,
            "structure": self.structure,
            "legs": [leg.canonical() for leg in self.legs],
            "quantity": int(self.quantity),
            "limit_price": _number(self.limit_price),
            "max_loss_usd": _number(self.max_loss_usd),
            "time_in_force": _enum_value(self.time_in_force),
            "thesis": self.thesis,
        }

    def content_hash(self) -> str:
        """SHA-256 of the canonical payload."""
        blob = json.dumps(self.canonical_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PortfolioState:
    """A point-in-time snapshot of the account, as the guard sees it."""

    equity: Decimal
    buying_power: Decimal
    open_positions: int
    positions_by_symbol: Mapping[str, int]
    portfolio_delta: float
    capital_at_risk_pct: float
    fetched_at: datetime


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """The risk guard's verdict on a proposal against a portfolio state."""

    approved: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OccContract:
    """The terms encoded in an OCC option symbol."""

    root: str
    expiration: date
    right: str  # "C" or "P"
    strike: Decimal


def parse_occ_symbol(symbol: str) -> OccContract:
    """Decode an OCC symbol, e.g. ``SPY260918P00753000``.

    The layout is root + YYMMDD + C/P + strike in thousandths. Parsing it
    means strikes and expiries are *derived* from the contracts actually being
    traded, rather than taken on trust from a separate proposal field.
    """
    tail = symbol[-15:]
    root = symbol[:-15]
    if not root or len(tail) != 15:
        raise ValueError(f"not an OCC symbol: {symbol!r}")

    right = tail[6]
    digits, strike_raw = tail[:6], tail[7:]
    if right not in ("C", "P") or not digits.isdigit() or not strike_raw.isdigit():
        raise ValueError(f"not an OCC symbol: {symbol!r}")

    return OccContract(
        root=root,
        expiration=date(2000 + int(digits[:2]), int(digits[2:4]), int(digits[4:6])),
        right=right,
        strike=Decimal(strike_raw) / 1000,
    )
