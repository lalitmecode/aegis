"""Deterministic enforcement of the mandate's numeric limits.

The critic argues about whether a trade is *sensible*. This module decides
whether it is *permitted*, using arithmetic only -- no judgement, no model, no
network. Given the same proposal, state and mandate it always returns the same
verdict, which is what makes it auditable.

Two principles shape it:

**Derive, do not trust.** A proposal states its own ``max_loss_usd``, but the
strikes in its OCC symbols say what the loss actually is. Where the structure
allows it, the guard computes max loss from the legs and uses the more
conservative of the two figures, reporting any disagreement.

**Fail closed.** Missing limits, an underivable structure, an absent delta, an
unverifiable market session -- each is a refusal, not a pass. A limit that
cannot be checked has not been satisfied.

Every violation is collected rather than short-circuited, so the audit log
records all the reasons a proposal was refused, not just the first.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from typing import Any, Mapping, Protocol
from zoneinfo import ZoneInfo

from aegis.core.proposal import (
    PortfolioState,
    RiskDecision,
    TradeProposal,
    parse_occ_symbol,
)

#: Shares per option contract. Deltas and strikes are scaled by this to reach
#: dollar and share-equivalent terms.
CONTRACT_MULTIPLIER = Decimal(100)

#: Dollar tolerance when comparing a claimed max loss against a derived one.
MAX_LOSS_TOLERANCE = Decimal("0.01")

#: Structures that open for a net credit; ``min_credit_to_max_loss_ratio``
#: only means something for these.
CREDIT_STRUCTURES = frozenset({"vertical_credit_spread", "iron_condor"})

EASTERN = ZoneInfo("America/New_York")


def _dec(value: Any) -> Decimal:
    return Decimal(str(value))


@dataclass(frozen=True, slots=True)
class SessionState:
    """What the market was doing at a given moment."""

    is_open: bool
    opened_at: datetime | None = None
    closes_at: datetime | None = None


class MarketSession(Protocol):
    """Source of market session facts, e.g. backed by Alpaca's clock/calendar."""

    def state(self, now: datetime) -> SessionState: ...


def derive_max_loss(proposal: TradeProposal) -> Decimal | None:
    """Compute worst-case loss in dollars from the legs themselves.

    Returns None when the structure is not one whose loss can be derived from
    strikes and net premium, which the guard treats as a refusal rather than
    as permission.
    """
    try:
        legs = [parse_occ_symbol(leg.symbol) for leg in proposal.legs]
    except ValueError:
        return None

    quantity = _dec(proposal.quantity)
    premium = _dec(proposal.limit_price) * CONTRACT_MULTIPLIER * quantity

    if proposal.structure == "vertical_debit_spread":
        # The most you can lose on a debit spread is what you paid.
        return premium

    if proposal.structure == "vertical_credit_spread":
        if len(legs) != 2 or legs[0].right != legs[1].right:
            return None
        width = abs(legs[0].strike - legs[1].strike)
        return width * CONTRACT_MULTIPLIER * quantity - premium

    if proposal.structure == "iron_condor":
        calls = [leg for leg in legs if leg.right == "C"]
        puts = [leg for leg in legs if leg.right == "P"]
        if len(calls) != 2 or len(puts) != 2:
            return None
        # Only one side can be breached, so the wider wing bounds the loss.
        width = max(
            abs(calls[0].strike - calls[1].strike),
            abs(puts[0].strike - puts[1].strike),
        )
        return width * CONTRACT_MULTIPLIER * quantity - premium

    return None


def net_delta(proposal: TradeProposal) -> Decimal | None:
    """Share-equivalent delta of the whole structure, or None if unknown.

    Long legs contribute positively, short legs negatively, and the result is
    scaled by ratio, quantity and the contract multiplier -- so a single short
    0.30-delta put reads as -30, not -0.30. ``max_portfolio_delta_abs: 100``
    is interpreted in these share-equivalent terms.
    """
    total = Decimal(0)
    for leg in proposal.legs:
        if leg.delta is None:
            return None
        sign = Decimal(-1) if leg.is_short else Decimal(1)
        total += _dec(leg.delta) * _dec(leg.ratio_qty) * sign
    return total * _dec(proposal.quantity) * CONTRACT_MULTIPLIER


class RiskGuard:
    """Checks a proposal against ``risk_limits``, ``strategy`` and ``timing``."""

    def __init__(
        self,
        mandate: Mapping[str, Any],
        *,
        session: MarketSession | None = None,
        clock=None,
    ) -> None:
        self._mandate = mandate
        self._session = session
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def evaluate(self, proposal: TradeProposal, state: PortfolioState) -> RiskDecision:
        """Return an approval verdict listing every limit the proposal breaches."""
        limits = self._mandate.get("risk_limits") or {}
        strategy = self._mandate.get("strategy") or {}
        if not limits:
            return RiskDecision(False, ("mandate has no risk_limits section",))

        now = self._clock()
        reasons: list[str] = []

        max_loss, loss_reasons = self._effective_max_loss(proposal)
        reasons += loss_reasons
        reasons += self._check_trade_size(max_loss, state, limits)
        reasons += self._check_position_counts(proposal, state, limits)
        reasons += self._check_delta(proposal, state, strategy, limits)
        reasons += self._check_capital(max_loss, state, limits)
        reasons += self._check_credit_ratio(proposal, max_loss, strategy)
        reasons += self._check_expiry(proposal, strategy, now)
        reasons += self._check_timing(now)

        return RiskDecision(approved=not reasons, reasons=tuple(reasons))

    # -- max loss ----------------------------------------------------------

    def _effective_max_loss(self, proposal: TradeProposal) -> tuple[Decimal, list[str]]:
        """The figure every dollar limit is checked against, plus any complaints."""
        claimed = _dec(proposal.max_loss_usd)
        derived = derive_max_loss(proposal)
        reasons: list[str] = []

        if derived is None:
            reasons.append(
                f"cannot derive max loss for structure {proposal.structure!r} "
                f"from its legs; refusing rather than trusting the claim"
            )
            return claimed, reasons

        if abs(derived - claimed) > MAX_LOSS_TOLERANCE:
            reasons.append(
                f"claimed max loss ${claimed} disagrees with ${derived} derived "
                f"from the legs"
            )
        # Take the worse of the two rather than picking a side.
        return max(derived, claimed), reasons

    # -- individual limits -------------------------------------------------

    def _check_trade_size(
        self, max_loss: Decimal, state: PortfolioState, limits: Mapping[str, Any]
    ) -> list[str]:
        reasons = []
        cap = limits.get("max_loss_per_trade_usd")
        if cap is not None and max_loss > _dec(cap):
            reasons.append(f"max loss ${max_loss} exceeds max_loss_per_trade_usd ${cap}")

        pct_cap = limits.get("max_loss_per_trade_pct_of_equity")
        if pct_cap is not None:
            equity = _dec(state.equity)
            if equity <= 0:
                reasons.append("equity is zero or negative; cannot size a trade against it")
            else:
                pct = max_loss / equity * 100
                if pct > _dec(pct_cap):
                    reasons.append(
                        f"max loss is {pct:.2f}% of equity, over "
                        f"max_loss_per_trade_pct_of_equity {pct_cap}%"
                    )
        return reasons

    def _check_position_counts(
        self, proposal: TradeProposal, state: PortfolioState, limits: Mapping[str, Any]
    ) -> list[str]:
        reasons = []
        cap = limits.get("max_open_positions")
        if cap is not None and state.open_positions + 1 > int(cap):
            reasons.append(
                f"would be position {state.open_positions + 1}, over "
                f"max_open_positions {cap}"
            )

        per_symbol = limits.get("max_positions_per_symbol")
        if per_symbol is not None:
            held = int(state.positions_by_symbol.get(proposal.underlying, 0))
            if held + 1 > int(per_symbol):
                reasons.append(
                    f"already holding {held} position(s) in {proposal.underlying}, "
                    f"over max_positions_per_symbol {per_symbol}"
                )
        return reasons

    def _check_delta(
        self,
        proposal: TradeProposal,
        state: PortfolioState,
        strategy: Mapping[str, Any],
        limits: Mapping[str, Any],
    ) -> list[str]:
        reasons = []
        short_cap = (strategy.get("delta_limits") or {}).get("short_leg_abs_delta_max")
        if short_cap is not None:
            for leg in proposal.legs:
                if not leg.is_short:
                    continue
                if leg.delta is None:
                    reasons.append(f"short leg {leg.symbol} has no delta; cannot check it")
                elif abs(_dec(leg.delta)) > _dec(short_cap):
                    reasons.append(
                        f"short leg {leg.symbol} delta {abs(_dec(leg.delta))} exceeds "
                        f"short_leg_abs_delta_max {short_cap}"
                    )

        portfolio_cap = limits.get("max_portfolio_delta_abs")
        if portfolio_cap is not None:
            delta = net_delta(proposal)
            if delta is None:
                reasons.append("a leg is missing delta; cannot check portfolio delta")
            else:
                projected = _dec(state.portfolio_delta) + delta
                if abs(projected) > _dec(portfolio_cap):
                    reasons.append(
                        f"portfolio delta would reach {projected}, over "
                        f"max_portfolio_delta_abs {portfolio_cap}"
                    )
        return reasons

    def _check_capital(
        self, max_loss: Decimal, state: PortfolioState, limits: Mapping[str, Any]
    ) -> list[str]:
        reasons = []
        cap = limits.get("max_total_capital_at_risk_pct")
        if cap is not None:
            equity = _dec(state.equity)
            if equity <= 0:
                reasons.append("equity is zero or negative; cannot compute capital at risk")
            else:
                projected = _dec(state.capital_at_risk_pct) + (max_loss / equity * 100)
                if projected > _dec(cap):
                    reasons.append(
                        f"capital at risk would reach {projected:.2f}%, over "
                        f"max_total_capital_at_risk_pct {cap}%"
                    )

        buffer_floor = limits.get("min_buying_power_buffer_usd")
        if buffer_floor is not None:
            remaining = _dec(state.buying_power) - max_loss
            if remaining < _dec(buffer_floor):
                reasons.append(
                    f"buying power would fall to ${remaining}, under "
                    f"min_buying_power_buffer_usd ${buffer_floor}"
                )
        return reasons

    def _check_credit_ratio(
        self, proposal: TradeProposal, max_loss: Decimal, strategy: Mapping[str, Any]
    ) -> list[str]:
        floor = strategy.get("min_credit_to_max_loss_ratio")
        if floor is None or proposal.structure not in CREDIT_STRUCTURES:
            return []
        if max_loss <= 0:
            return [f"max loss is ${max_loss}; cannot compute a credit ratio"]

        credit = _dec(proposal.limit_price) * CONTRACT_MULTIPLIER * _dec(proposal.quantity)
        ratio = credit / max_loss
        if ratio < _dec(floor):
            return [
                f"credit/max-loss ratio {ratio:.3f} is under "
                f"min_credit_to_max_loss_ratio {floor}"
            ]
        return []

    def _check_expiry(
        self, proposal: TradeProposal, strategy: Mapping[str, Any], now: datetime
    ) -> list[str]:
        window = strategy.get("expiry_window_days") or {}
        if not window:
            return []

        try:
            expirations = {parse_occ_symbol(leg.symbol).expiration for leg in proposal.legs}
        except ValueError as exc:
            return [f"cannot read expiry from legs: {exc}"]

        if len(expirations) != 1:
            return [
                "legs span multiple expiries "
                f"({', '.join(sorted(str(e) for e in expirations))}); not a permitted structure"
            ]

        expiration = expirations.pop()
        dte = (expiration - now.astimezone(timezone.utc).date()).days
        low, high = window.get("min"), window.get("max")
        if low is not None and dte < int(low):
            return [f"expiry is {dte} days out, under expiry_window_days.min {low}"]
        if high is not None and dte > int(high):
            return [f"expiry is {dte} days out, over expiry_window_days.max {high}"]
        return []

    def _check_timing(self, now: datetime) -> list[str]:
        timing = self._mandate.get("timing") or {}
        if not timing:
            return []

        needs_session = (
            timing.get("market_hours_only")
            or timing.get("no_new_positions_within_minutes_of_open") is not None
        )
        reasons: list[str] = []

        if needs_session and self._session is None:
            return ["no market session source configured; cannot verify trading hours"]

        if needs_session:
            session = self._session.state(now)
            if timing.get("market_hours_only") and not session.is_open:
                reasons.append("market is closed and mandate sets market_hours_only")

            settle = timing.get("no_new_positions_within_minutes_of_open")
            if settle is not None:
                if session.opened_at is None:
                    reasons.append("market open time unknown; cannot honour the opening delay")
                elif session.opened_at <= now < session.opened_at + timedelta(
                    minutes=int(settle)
                ):
                    reasons.append(
                        f"within {settle} minutes of the open "
                        f"(opened {session.opened_at.isoformat()})"
                    )

        cutoff = timing.get("no_new_positions_after")
        if cutoff:
            hour, _, minute = str(cutoff).partition(":")
            local = now.astimezone(EASTERN)
            if local.timetz().replace(tzinfo=None) >= time(int(hour), int(minute or 0)):
                reasons.append(
                    f"local time {local:%H:%M} is past no_new_positions_after {cutoff} ET"
                )
        return reasons
