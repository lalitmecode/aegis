"""The agent that proposes trades.

The design principle here is that **the LLM writes the reasoning and the code
makes the decisions**. Strikes, widths, sizing, and structure are derived from
the observed option chain and the mandate's numeric limits by ordinary Python.
The model is asked for one thing only: a short written thesis explaining the
trade that has already been chosen.

That split is deliberate. A hallucinating model produces a bad *explanation*,
which a human reads and rejects. It cannot produce a bad *trade*, because it is
never given the chance to pick one. The thesis is hashed into the proposal, so
the human approves the reasoning along with the position.

Every field on the returned proposal comes from the chain -- deltas, strikes,
the limit price from the midpoint. The observation verifier re-checks all of
them before execution, so populating them from anything other than live data
would only guarantee a refusal later.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Mapping

from alpaca.trading.enums import OrderSide, PositionIntent

from aegis.core.option_chain import fetch_chain
from aegis.core.proposal import OrderLeg, PortfolioState, TradeProposal

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"
CONTRACT_MULTIPLIER = Decimal(100)
DEFAULT_TARGET_DTE = 30
DEFAULT_MONEYNESS = 0.05

#: Candidate spread widths, in points of underlying price.
#:
#: Strike ladders are not a common unit: "two strikes out" is 2 points on SPY's
#: 1-point ladder and 10 points on a 5-point ladder, so counting strikes makes
#: the width of the risk an accident of the chain's granularity.
DEFAULT_WIDTHS = (Decimal(5), Decimal(10))

#: How far the chosen long strike may sit from the requested width before the
#: candidate is discarded as not really that spread.
_WIDTH_TOLERANCE = Decimal("0.5")

_THESIS_SYSTEM = """You are the research analyst for Aegis, a mandate-governed \
options trading agent.

The trade below has already been selected. Deterministic code chose the strikes, \
the width, the expiry and the size from live option-chain data and the fund's \
numeric risk limits. You cannot change any of it, and nothing you write will \
alter the order that gets placed.

Your only job is the thesis: a short written rationale a human will read when \
deciding whether to approve this trade. Explain what the position expresses and \
what would have to happen for it to lose. Be concrete about the numbers you are \
given. If the trade looks weak on its own terms, say so plainly -- a thesis that \
talks a human out of a trade is a useful thesis.

Respond with two to four sentences of plain prose. Do not use headings, bullet \
points, or a preamble such as "Here is the thesis"."""


def _dec(value: Any) -> Decimal:
    return Decimal(str(value))


@dataclass(frozen=True, slots=True)
class _Spread:
    """A candidate structure, before sizing."""

    short: Any
    long: Any
    width: Decimal
    credit: Decimal  # net credit per share
    per_contract_loss: Decimal
    per_contract_delta: Decimal


class ResearchAgent:
    """Proposes one defined-risk spread, or nothing."""

    def __init__(
        self,
        mandate: Mapping[str, Any],
        chain_fetcher=fetch_chain,
        llm_client: Any | None = None,
        *,
        model: str = DEFAULT_MODEL,
        contract_type: str = "put",
        widths: tuple[Decimal, ...] = DEFAULT_WIDTHS,
        target_dte: int = DEFAULT_TARGET_DTE,
        moneyness: float = DEFAULT_MONEYNESS,
        today=None,
    ) -> None:
        self._mandate = mandate
        self._fetch = chain_fetcher
        self._llm = llm_client
        self._model = model
        self._contract_type = contract_type
        self._widths = tuple(_dec(w) for w in widths)
        self._target_dte = target_dte
        self._moneyness = moneyness
        self._today = today or date.today

    def propose(self, underlying: str, state: PortfolioState) -> TradeProposal | None:
        """Return a compliant spread, or None if no such spread exists.

        "No trade" is a correct and expected outcome. Every path that returns
        None logs why.
        """
        universe = self._mandate.get("universe") or {}
        limits = self._mandate.get("risk_limits") or {}
        strategy = self._mandate.get("strategy") or {}

        if underlying not in (universe.get("allowed_symbols") or []):
            return self._no_trade(underlying, "not in mandate universe.allowed_symbols")
        if not limits or not strategy:
            return self._no_trade(underlying, "mandate is missing risk_limits or strategy")

        capacity = self._position_capacity(underlying, state, limits)
        if capacity is not None:
            return self._no_trade(underlying, capacity)

        window = strategy.get("expiry_window_days") or {}
        expiry_window = (int(window.get("min", 7)), int(window.get("max", 45)))
        try:
            chain = self._fetch(
                underlying,
                target_dte=self._target_dte,
                moneyness=self._moneyness,
                expiry_window=expiry_window,
                contract_type=self._contract_type,
                today=self._today(),
            )
        except LookupError as exc:
            return self._no_trade(underlying, f"no chain available: {exc}")

        if self._min_open_interest() is None:
            return self._no_trade(
                underlying, "mandate does not define universe.min_open_interest"
            )

        spread = self._select_spread(chain, strategy)
        if spread is None:
            return self._no_trade(
                underlying,
                "no strike pair satisfies the delta ceiling, liquidity floor, "
                "and credit ratio",
            )

        quantity = self._size(spread, state, limits)
        if quantity < 1:
            return self._no_trade(
                underlying,
                f"a single {spread.width}-wide spread (${spread.per_contract_loss} "
                f"max loss) does not fit the risk limits",
            )

        proposal = self._build(underlying, chain, spread, quantity)
        thesis = self._write_thesis(proposal, chain, spread)
        proposal = self._with_thesis(proposal, thesis)

        log.info(
            "proposing %s %s %s/%s x%d for %s credit (max loss $%s)",
            proposal.proposal_id,
            underlying,
            spread.short.strike,
            spread.long.strike,
            quantity,
            proposal.limit_price,
            proposal.max_loss_usd,
        )
        return proposal

    def _no_trade(self, underlying: str, reason: str) -> None:
        """Log why nothing was proposed and return None. Not an error path."""
        log.info("no trade for %s: %s", underlying, reason)
        return None

    # -- selection ---------------------------------------------------------

    def _position_capacity(self, underlying, state, limits) -> str | None:
        """Reason the portfolio has no room, or None if it does."""
        cap = limits.get("max_open_positions")
        if cap is not None and state.open_positions + 1 > int(cap):
            return f"already holding {state.open_positions} of {cap} permitted positions"

        per_symbol = limits.get("max_positions_per_symbol")
        if per_symbol is not None:
            held = int(state.positions_by_symbol.get(underlying, 0))
            if held + 1 > int(per_symbol):
                return f"already holding {held} position(s) in {underlying}"
        return None

    def _select_spread(self, chain, strategy) -> _Spread | None:
        """Pick the short leg by delta, then define the risk with a long leg."""
        delta_cap = _dec((strategy.get("delta_limits") or {}).get("short_leg_abs_delta_max", 1))
        min_ratio = strategy.get("min_credit_to_max_loss_ratio")
        min_oi = self._min_open_interest()
        if min_oi is None:
            return None  # caller reports it; see propose()

        ladder = self._otm_ladder(chain)
        if not ladder:
            return None

        anchor = chain.nearest_delta(delta_cap, self._contract_type)
        start = self._index_of(ladder, anchor) if anchor else len(ladder) - 1

        # Walk further out-of-the-money until the short leg clears the mandate.
        for i in range(start, -1, -1):
            short = ladder[i]
            if not self._eligible(short, min_oi) or abs(_dec(short.delta)) > delta_cap:
                continue
            candidates = [
                self._build_spread(short, long)
                for long in (
                    self._leg_at_width(ladder, i, width, min_oi) for width in self._widths
                )
                if long is not None
            ]
            viable = [
                s
                for s in candidates
                if s is not None
                and s.credit > 0
                and (min_ratio is None or s.credit * CONTRACT_MULTIPLIER / s.per_contract_loss >= _dec(min_ratio))
            ]
            if viable:
                # Capital preservation takes precedence: narrowest risk wins.
                return min(viable, key=lambda s: s.per_contract_loss)
        return None

    def _otm_ladder(self, chain) -> list:
        """Contracts of the traded type, furthest out-of-the-money first."""
        legs = [c for c in chain.contracts if c.type == self._contract_type]
        return sorted(legs, key=lambda c: c.strike, reverse=self._contract_type == "call")

    def _leg_at_width(self, ladder, short_index: int, width: Decimal, min_oi: int):
        """The listed strike closest to ``width`` points further out of the money.

        Returns None when the chain lists nothing near that distance, rather
        than silently substituting a spread of a different width.
        """
        short = ladder[short_index]
        sign = -1 if self._contract_type == "put" else 1
        target = _dec(short.strike) + sign * width

        candidates = [c for c in ladder[:short_index] if self._eligible(c, min_oi)]
        if not candidates:
            return None
        nearest = min(candidates, key=lambda c: abs(_dec(c.strike) - target))
        if abs(_dec(nearest.strike) - target) > width * _WIDTH_TOLERANCE:
            return None
        return nearest

    @staticmethod
    def _index_of(ladder, contract) -> int:
        for i, candidate in enumerate(ladder):
            if candidate.symbol == contract.symbol:
                return i
        return len(ladder) - 1

    def _min_open_interest(self) -> int | None:
        """The mandate's liquidity floor, or None if it does not state one."""
        floor = (self._mandate.get("universe") or {}).get("min_open_interest")
        return None if floor is None else int(floor)

    @staticmethod
    def _eligible(contract, min_oi: int) -> bool:
        return (
            contract.delta is not None
            and contract.mid is not None
            and contract.open_interest is not None
            and contract.open_interest >= min_oi
        )

    def _build_spread(self, short, long) -> _Spread | None:
        width = abs(_dec(short.strike) - _dec(long.strike))
        if width <= 0:
            return None
        credit = (_dec(short.mid) - _dec(long.mid)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        per_contract_loss = width * CONTRACT_MULTIPLIER - credit * CONTRACT_MULTIPLIER
        if per_contract_loss <= 0:
            return None
        # Short leg is sold, long leg bought: share-equivalent delta per spread.
        per_contract_delta = (
            _dec(long.delta) - _dec(short.delta)
        ) * CONTRACT_MULTIPLIER
        return _Spread(short, long, width, credit, per_contract_loss, per_contract_delta)

    # -- sizing ------------------------------------------------------------

    def _size(self, spread: _Spread, state: PortfolioState, limits) -> int:
        """Largest quantity that satisfies every dollar and delta limit."""
        loss = spread.per_contract_loss
        equity = _dec(state.equity)
        caps: list[Decimal] = []

        usd_cap = limits.get("max_loss_per_trade_usd")
        if usd_cap is not None:
            caps.append(_dec(usd_cap) / loss)

        pct_cap = limits.get("max_loss_per_trade_pct_of_equity")
        if pct_cap is not None and equity > 0:
            caps.append(equity * _dec(pct_cap) / 100 / loss)

        risk_cap = limits.get("max_total_capital_at_risk_pct")
        if risk_cap is not None and equity > 0:
            headroom = (_dec(risk_cap) - _dec(state.capital_at_risk_pct)) / 100 * equity
            caps.append(headroom / loss)

        buffer_floor = limits.get("min_buying_power_buffer_usd")
        if buffer_floor is not None:
            caps.append((_dec(state.buying_power) - _dec(buffer_floor)) / loss)

        if not caps:
            return 0
        quantity = int(min(caps))  # floor
        return max(0, self._fit_portfolio_delta(quantity, spread, state, limits))

    def _fit_portfolio_delta(self, quantity, spread, state, limits) -> int:
        """Step the size down until projected portfolio delta fits."""
        cap = limits.get("max_portfolio_delta_abs")
        if cap is None:
            return quantity
        current = _dec(state.portfolio_delta)
        for n in range(quantity, 0, -1):
            if abs(current + spread.per_contract_delta * n) <= _dec(cap):
                return n
        return 0

    # -- assembly ----------------------------------------------------------

    def _build(self, underlying, chain, spread: _Spread, quantity: int) -> TradeProposal:
        return TradeProposal(
            proposal_id=self._proposal_id(underlying, chain, spread),
            underlying=underlying,
            structure="vertical_credit_spread",
            legs=(
                OrderLeg(
                    spread.short.symbol,
                    OrderSide.SELL,
                    PositionIntent.SELL_TO_OPEN,
                    delta=_dec(spread.short.delta),
                ),
                OrderLeg(
                    spread.long.symbol,
                    OrderSide.BUY,
                    PositionIntent.BUY_TO_OPEN,
                    delta=_dec(spread.long.delta),
                ),
            ),
            quantity=quantity,
            limit_price=spread.credit,
            max_loss_usd=spread.per_contract_loss * quantity,
        )

    @staticmethod
    def _proposal_id(underlying, chain, spread: _Spread) -> str:
        return (
            f"{underlying.lower()}-{chain.expiration:%Y%m%d}"
            f"-{spread.short.strike:.0f}-{spread.long.strike:.0f}"
        )

    @staticmethod
    def _with_thesis(proposal: TradeProposal, thesis: str | None) -> TradeProposal:
        from dataclasses import replace

        return replace(proposal, thesis=thesis)

    # -- the one thing the model is asked for ------------------------------

    def _write_thesis(self, proposal, chain, spread: _Spread) -> str | None:
        """Ask Claude for the rationale. A failure here costs the explanation, not the trade."""
        if self._llm is None:
            log.info("no LLM client configured; proposing %s without a thesis", proposal.proposal_id)
            return None

        facts = (
            f"Underlying: {proposal.underlying} at {chain.spot}\n"
            f"Structure: {proposal.structure}, {proposal.quantity} contract(s)\n"
            f"Expiry: {chain.expiration} ({chain.dte} days out)\n"
            f"Short leg: {spread.short.symbol} strike {spread.short.strike}, "
            f"delta {spread.short.delta}, IV {spread.short.implied_volatility}, "
            f"open interest {spread.short.open_interest}\n"
            f"Long leg: {spread.long.symbol} strike {spread.long.strike}, "
            f"delta {spread.long.delta}, open interest {spread.long.open_interest}\n"
            f"Net credit: {proposal.limit_price} per spread\n"
            f"Width: {spread.width} points\n"
            f"Max loss: ${proposal.max_loss_usd} total\n"
            f"Breakeven: {_dec(spread.short.strike) - spread.credit}\n"
        )

        try:
            response = self._llm.messages.create(
                model=self._model,
                max_tokens=1024,
                system=_THESIS_SYSTEM,
                messages=[{"role": "user", "content": facts}],
            )
        except Exception as exc:  # the trade does not depend on the model
            log.warning("thesis generation failed (%s); proposing without one", exc)
            return None

        thesis = _text_of(response).strip()
        return thesis or None


def _text_of(response: Any) -> str:
    """Concatenate text blocks, skipping thinking and other block types."""
    blocks = getattr(response, "content", None) or []
    return "".join(
        getattr(block, "text", "") for block in blocks if getattr(block, "type", None) == "text"
    )
