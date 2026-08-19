"""Reconciles what an agent *claims* about its legs against what the market shows.

The risk guard is deliberately pure: given a proposal and a state it does
arithmetic and nothing else. That purity has a cost -- it enforces limits over
figures the agent supplied about itself. This layer closes that gap, and it is
the only part of the pre-trade path that touches the network.

The design point that matters: when a claim and an observation disagree, the
proposal is **refused**, not corrected. Silently substituting the observed
value would let a misreporting agent keep trading while the audit trail records
a clean submission. The observed values are still returned, so an operator can
see what the truth was and re-propose against it -- but that is a new proposal,
with a new hash, needing new approval.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any, Callable, Mapping

from aegis.core.option_chain import fetch_chain
from aegis.core.proposal import TradeProposal, parse_occ_symbol

#: Absolute delta difference tolerated between claim and observation.
DEFAULT_TOLERANCE = Decimal("0.02")

#: Initial strike band. Widened once if the proposal's strikes fall outside it.
_PROBE_MONEYNESS = 0.10


@dataclass(frozen=True, slots=True)
class Discrepancy:
    """One way in which the proposal disagrees with the market."""

    symbol: str
    field: str
    claimed: Any
    observed: Any
    detail: str

    def __str__(self) -> str:
        return f"{self.symbol} {self.field}: {self.detail}"


@dataclass(frozen=True, slots=True)
class ObservedLeg:
    """What the market actually shows for a leg."""

    symbol: str
    strike: Decimal | None = None
    delta: Decimal | None = None
    implied_volatility: Decimal | None = None
    bid: Decimal | None = None
    ask: Decimal | None = None
    open_interest: int | None = None
    listed: bool = True


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Outcome of reconciling a proposal against observed market data."""

    verified: bool
    discrepancies: tuple[Discrepancy, ...] = ()
    observed_legs: tuple[ObservedLeg, ...] = ()

    def summary(self) -> str:
        return "; ".join(str(d) for d in self.discrepancies)

    def as_audit_payload(self) -> list[dict[str, Any]]:
        return [
            {
                "symbol": d.symbol,
                "field": d.field,
                "claimed": None if d.claimed is None else str(d.claimed),
                "observed": None if d.observed is None else str(d.observed),
                "detail": d.detail,
            }
            for d in self.discrepancies
        ]


def _dec(value: Any) -> Decimal | None:
    return None if value is None else Decimal(str(value))


class ObservationVerifier:
    """Checks each leg of a proposal against a freshly fetched option chain."""

    def __init__(
        self,
        chain_fetcher: Callable[..., Any] = fetch_chain,
        tolerance: Decimal = DEFAULT_TOLERANCE,
        *,
        min_open_interest: int,
        today: Callable[[], date] | None = None,
    ) -> None:
        """``min_open_interest`` has no default on purpose.

        The liquidity floor is a mandate decision, not a library constant --
        see :meth:`from_mandate`.
        """
        self._fetch = chain_fetcher
        self._tolerance = Decimal(str(tolerance))
        self._min_open_interest = int(min_open_interest)
        self._today = today or date.today

    @classmethod
    def from_mandate(
        cls,
        mandate: Mapping[str, Any],
        chain_fetcher: Callable[..., Any] = fetch_chain,
        tolerance: Decimal = DEFAULT_TOLERANCE,
        *,
        today: Callable[[], date] | None = None,
    ) -> "ObservationVerifier":
        """Build a verifier whose liquidity floor comes from the mandate.

        Raises:
            ValueError: the mandate does not define ``universe.min_open_interest``.
                Guessing a floor here would be inventing policy.
        """
        floor = (mandate.get("universe") or {}).get("min_open_interest")
        if floor is None:
            raise ValueError("mandate does not define universe.min_open_interest")
        return cls(
            chain_fetcher,
            tolerance,
            min_open_interest=int(floor),
            today=today,
        )

    def verify(self, proposal: TradeProposal) -> VerificationResult:
        """Reconcile every leg. Any discrepancy is blocking."""
        today = self._today()
        discrepancies: list[Discrepancy] = []
        observed: list[ObservedLeg] = []

        # One chain fetch per distinct expiry, not per leg.
        by_expiry: dict[date, list] = {}
        for leg in proposal.legs:
            try:
                contract = parse_occ_symbol(leg.symbol)
            except ValueError as exc:
                discrepancies.append(
                    Discrepancy(leg.symbol, "symbol", leg.symbol, None, str(exc))
                )
                observed.append(ObservedLeg(leg.symbol, listed=False))
                continue
            by_expiry.setdefault(contract.expiration, []).append((leg, contract))

        for expiration, entries in by_expiry.items():
            strikes = [contract.strike for _, contract in entries]
            try:
                chain = self._chain_for(proposal.underlying, expiration, strikes, today)
            except LookupError as exc:
                for leg, _ in entries:
                    discrepancies.append(
                        Discrepancy(
                            leg.symbol,
                            "listing",
                            None,
                            None,
                            f"no chain available for {expiration}: {exc}",
                        )
                    )
                    observed.append(ObservedLeg(leg.symbol, listed=False))
                continue

            listed = {contract.symbol: contract for contract in chain.contracts}
            for leg, parsed in entries:
                leg_observed, leg_discrepancies = self._check_leg(leg, parsed, listed)
                observed.append(leg_observed)
                discrepancies.extend(leg_discrepancies)

        return VerificationResult(
            verified=not discrepancies,
            discrepancies=tuple(discrepancies),
            observed_legs=tuple(observed),
        )

    # -- per-leg checks ----------------------------------------------------

    def _check_leg(self, leg, parsed, listed) -> tuple[ObservedLeg, list[Discrepancy]]:
        found = listed.get(leg.symbol)
        if found is None:
            return (
                ObservedLeg(leg.symbol, strike=parsed.strike, listed=False),
                [
                    Discrepancy(
                        leg.symbol,
                        "listing",
                        f"strike {parsed.strike}",
                        None,
                        "strike is not listed in the observed chain",
                    )
                ],
            )

        observed = ObservedLeg(
            symbol=leg.symbol,
            strike=_dec(found.strike),
            delta=_dec(found.delta),
            implied_volatility=_dec(found.implied_volatility),
            bid=_dec(found.bid),
            ask=_dec(found.ask),
            open_interest=found.open_interest,
        )
        discrepancies: list[Discrepancy] = []
        discrepancies += self._check_quote(leg, observed)
        discrepancies += self._check_delta(leg, observed)
        discrepancies += self._check_open_interest(leg, observed)
        return observed, discrepancies

    def _check_quote(self, leg, observed: ObservedLeg) -> list[Discrepancy]:
        if observed.bid is None or observed.ask is None:
            return [
                Discrepancy(
                    leg.symbol,
                    "quote",
                    None,
                    f"bid={observed.bid} ask={observed.ask}",
                    "no two-sided quote; the leg is not reliably tradable",
                )
            ]
        return []

    def _check_delta(self, leg, observed: ObservedLeg) -> list[Discrepancy]:
        claimed = _dec(leg.delta)
        if claimed is None:
            return [
                Discrepancy(
                    leg.symbol,
                    "delta",
                    None,
                    observed.delta,
                    "proposal claims no delta for this leg",
                )
            ]
        if observed.delta is None:
            return [
                Discrepancy(
                    leg.symbol,
                    "delta",
                    claimed,
                    None,
                    "chain reports no delta for this leg",
                )
            ]

        drift = abs(claimed - observed.delta)
        if drift > self._tolerance:
            return [
                Discrepancy(
                    leg.symbol,
                    "delta",
                    claimed,
                    observed.delta,
                    f"claimed {claimed} but observed {observed.delta} "
                    f"(off by {drift}, tolerance {self._tolerance})",
                )
            ]
        return []

    def _check_open_interest(self, leg, observed: ObservedLeg) -> list[Discrepancy]:
        if observed.open_interest is None:
            return [
                Discrepancy(
                    leg.symbol,
                    "open_interest",
                    None,
                    None,
                    "chain reports no open interest; liquidity floor unverifiable",
                )
            ]
        if observed.open_interest < self._min_open_interest:
            return [
                Discrepancy(
                    leg.symbol,
                    "open_interest",
                    None,
                    observed.open_interest,
                    f"open interest {observed.open_interest} is under the "
                    f"{self._min_open_interest} floor",
                )
            ]
        return []

    # -- chain retrieval ---------------------------------------------------

    def _chain_for(self, underlying: str, expiration: date, strikes, today: date):
        """Fetch the chain for one exact expiry, widened to cover every strike."""
        dte = (expiration - today).days
        chain = self._fetch(
            underlying,
            target_dte=dte,
            moneyness=_PROBE_MONEYNESS,
            expiry_window=(dte, dte),
            today=today,
        )

        spot = _dec(chain.spot) or Decimal(0)
        if spot <= 0 or not strikes:
            return chain

        needed = max(abs(Decimal(str(k)) - spot) for k in strikes) / spot
        if needed >= Decimal(str(_PROBE_MONEYNESS)):
            chain = self._fetch(
                underlying,
                target_dte=dte,
                moneyness=float(needed * Decimal("1.05")),
                expiry_window=(dte, dte),
                today=today,
            )
        return chain
