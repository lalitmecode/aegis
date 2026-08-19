"""Verification tests. The chain fetcher is stubbed -- nothing here hits Alpaca."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import Decimal

import pytest
from alpaca.trading.enums import OrderSide, PositionIntent

from aegis.core.option_chain import Chain, Contract
from aegis.core.proposal import OrderLeg, TradeProposal
from aegis.core.verifier import ObservationVerifier, VerificationResult

TODAY = date(2026, 8, 19)
EXPIRY = date(2026, 9, 18)
SHORT = "SPY260918P00753000"
LONG = "SPY260918P00748000"


def contract(symbol, strike, delta, open_interest=2804, bid=6.50, ask=6.79):
    return Contract(
        symbol=symbol,
        type="put",
        strike=strike,
        expiration=EXPIRY,
        bid=bid,
        ask=ask,
        delta=delta,
        implied_volatility=0.154,
        open_interest=open_interest,
    )


def chain(*contracts, spot=767.35):
    return Chain(
        underlying="SPY",
        spot=spot,
        expiration=EXPIRY,
        dte=30,
        contracts=tuple(contracts),
    )


class StubFetcher:
    """Stands in for fetch_chain, recording how it was called."""

    def __init__(self, *chains):
        self.chains = list(chains)
        self.calls: list[dict] = []

    def __call__(self, underlying, **kwargs):
        self.calls.append({"underlying": underlying, **kwargs})
        return self.chains[min(len(self.calls) - 1, len(self.chains) - 1)]


def build_verifier(fetcher, **kw) -> ObservationVerifier:
    kw.setdefault("min_open_interest", 500)
    return ObservationVerifier(fetcher, today=lambda: TODAY, **kw)


def proposal_with(short_delta=Decimal("-0.30"), long_delta=Decimal("-0.22")):
    return TradeProposal(
        proposal_id="prop-001",
        underlying="SPY",
        structure="vertical_credit_spread",
        legs=(
            OrderLeg(SHORT, OrderSide.SELL, PositionIntent.SELL_TO_OPEN, delta=short_delta),
            OrderLeg(LONG, OrderSide.BUY, PositionIntent.BUY_TO_OPEN, delta=long_delta),
        ),
        quantity=1,
        limit_price=Decimal("1.10"),
        max_loss_usd=Decimal("390"),
    )


HONEST_CHAIN = chain(
    contract(SHORT, Decimal("753"), -0.3014),
    contract(LONG, Decimal("748"), -0.2210),
)


# --------------------------------------------------------------------------
# truthful proposals
# --------------------------------------------------------------------------


def test_a_truthful_proposal_verifies():
    result = build_verifier(StubFetcher(HONEST_CHAIN)).verify(proposal_with())
    assert result.verified, result.summary()
    assert result.discrepancies == ()


def test_observed_values_are_returned_even_when_everything_matches():
    result = build_verifier(StubFetcher(HONEST_CHAIN)).verify(proposal_with())
    observed = {leg.symbol: leg for leg in result.observed_legs}
    assert observed[SHORT].delta == Decimal("-0.3014")
    assert observed[SHORT].open_interest == 2804
    assert observed[LONG].strike == Decimal("748")


def test_drift_inside_tolerance_is_accepted():
    # Claimed -0.30, observed -0.3014: 0.0014 of drift, well inside 0.02.
    assert build_verifier(StubFetcher(HONEST_CHAIN)).verify(proposal_with()).verified


def test_drift_just_outside_tolerance_is_refused():
    result = build_verifier(StubFetcher(HONEST_CHAIN)).verify(
        proposal_with(short_delta=Decimal("-0.33"))
    )
    assert not result.verified
    assert result.discrepancies[0].field == "delta"


# --------------------------------------------------------------------------
# lying proposals
# --------------------------------------------------------------------------


def test_a_lying_delta_is_refused():
    """The agent claims a 0.10-delta short leg; the market says 0.45."""
    result = build_verifier(StubFetcher(
        chain(contract(SHORT, Decimal("753"), -0.45), contract(LONG, Decimal("748"), -0.22))
    )).verify(proposal_with(short_delta=Decimal("-0.10")))

    assert not result.verified
    bad = result.discrepancies[0]
    assert bad.field == "delta"
    assert bad.symbol == SHORT
    assert bad.claimed == Decimal("-0.10")
    assert bad.observed == Decimal("-0.45")


def test_observed_values_are_not_substituted_on_a_discrepancy():
    """The corrected value is reported, never quietly swapped in."""
    lying = proposal_with(short_delta=Decimal("-0.10"))
    result = build_verifier(StubFetcher(
        chain(contract(SHORT, Decimal("753"), -0.45), contract(LONG, Decimal("748"), -0.22))
    )).verify(lying)

    assert not result.verified
    # The proposal is untouched: it still carries the claim it made.
    assert lying.legs[0].delta == Decimal("-0.10")
    # And the truth is available for an operator to re-propose against.
    observed = {leg.symbol: leg for leg in result.observed_legs}
    assert observed[SHORT].delta == Decimal("-0.45")


def test_an_illiquid_leg_is_refused():
    """Closes the mandate's 500-open-interest floor, which nothing else checks."""
    result = build_verifier(StubFetcher(
        chain(
            contract(SHORT, Decimal("753"), -0.3014, open_interest=12),
            contract(LONG, Decimal("748"), -0.2210),
        )
    )).verify(proposal_with())

    assert not result.verified
    bad = result.discrepancies[0]
    assert bad.field == "open_interest"
    assert bad.observed == 12
    assert "500" in bad.detail


def test_open_interest_exactly_at_the_floor_passes():
    result = build_verifier(StubFetcher(
        chain(
            contract(SHORT, Decimal("753"), -0.3014, open_interest=500),
            contract(LONG, Decimal("748"), -0.2210),
        )
    )).verify(proposal_with())
    assert result.verified, result.summary()


def test_an_unlisted_strike_is_refused():
    result = build_verifier(StubFetcher(chain(contract(LONG, Decimal("748"), -0.221)))).verify(
        proposal_with()
    )
    assert not result.verified
    assert result.discrepancies[0].field == "listing"
    assert result.discrepancies[0].symbol == SHORT


def test_a_leg_without_a_two_sided_quote_is_refused():
    result = build_verifier(StubFetcher(
        chain(
            contract(SHORT, Decimal("753"), -0.3014, bid=None),
            contract(LONG, Decimal("748"), -0.2210),
        )
    )).verify(proposal_with())
    assert not result.verified
    assert result.discrepancies[0].field == "quote"


def test_a_leg_claiming_no_delta_is_refused():
    result = build_verifier(StubFetcher(HONEST_CHAIN)).verify(
        proposal_with(short_delta=None)
    )
    assert not result.verified
    assert result.discrepancies[0].field == "delta"


def test_missing_open_interest_is_refused_rather_than_assumed_liquid():
    result = build_verifier(StubFetcher(
        chain(
            contract(SHORT, Decimal("753"), -0.3014, open_interest=None),
            contract(LONG, Decimal("748"), -0.2210),
        )
    )).verify(proposal_with())
    assert not result.verified
    assert result.discrepancies[0].field == "open_interest"


def test_every_bad_leg_is_reported_not_just_the_first():
    result = build_verifier(StubFetcher(
        chain(
            contract(SHORT, Decimal("753"), -0.45, open_interest=12),
            contract(LONG, Decimal("748"), -0.99),
        )
    )).verify(proposal_with())
    assert len({d.symbol for d in result.discrepancies}) == 2


# --------------------------------------------------------------------------
# chain retrieval
# --------------------------------------------------------------------------


def test_one_fetch_per_expiry_not_per_leg():
    fetcher = StubFetcher(HONEST_CHAIN)
    build_verifier(fetcher).verify(proposal_with())
    assert len(fetcher.calls) == 1


def test_the_exact_expiry_is_requested():
    fetcher = StubFetcher(HONEST_CHAIN)
    build_verifier(fetcher).verify(proposal_with())
    call = fetcher.calls[0]
    assert call["underlying"] == "SPY"
    assert call["target_dte"] == 30           # 2026-08-19 -> 2026-09-18
    assert call["expiry_window"] == (30, 30)  # pinned, not "nearest"


def test_the_strike_band_widens_when_legs_sit_far_from_spot():
    far = chain(
        contract(SHORT, Decimal("753"), -0.3014),
        contract(LONG, Decimal("748"), -0.2210),
        spot=1000.0,  # legs are ~25% away, outside the 10% probe band
    )
    fetcher = StubFetcher(far)
    build_verifier(fetcher).verify(proposal_with())

    assert len(fetcher.calls) == 2
    assert fetcher.calls[0]["moneyness"] == 0.10
    assert fetcher.calls[1]["moneyness"] > 0.25


def test_an_unavailable_chain_is_a_discrepancy_not_a_crash():
    class Missing:
        def __call__(self, underlying, **kwargs):
            raise LookupError("no contracts listed")

    result = build_verifier(Missing()).verify(proposal_with())
    assert not result.verified
    assert all(d.field == "listing" for d in result.discrepancies)
    assert len(result.discrepancies) == 2


def test_a_malformed_leg_symbol_is_reported():
    bad = replace(
        proposal_with(),
        legs=(replace(proposal_with().legs[0], symbol="NOT-AN-OCC-SYMBOL"),),
    )
    result = build_verifier(StubFetcher(HONEST_CHAIN)).verify(bad)
    assert not result.verified
    assert result.discrepancies[0].field == "symbol"


def test_audit_payload_is_json_friendly():
    result = build_verifier(StubFetcher(
        chain(contract(SHORT, Decimal("753"), -0.45), contract(LONG, Decimal("748"), -0.22))
    )).verify(proposal_with(short_delta=Decimal("-0.10")))

    import json

    payload = result.as_audit_payload()
    assert json.loads(json.dumps(payload))[0]["claimed"] == "-0.10"


# --------------------------------------------------------------------------
# the liquidity floor is mandate policy, not a library constant
# --------------------------------------------------------------------------


def test_from_mandate_reads_the_floor():
    import pathlib

    import yaml

    mandate = yaml.safe_load(
        (pathlib.Path(__file__).resolve().parent.parent / "config" / "mandate.yaml").read_text()
    )
    verifier = ObservationVerifier.from_mandate(
        mandate, StubFetcher(HONEST_CHAIN), today=lambda: TODAY
    )
    assert verifier._min_open_interest == mandate["universe"]["min_open_interest"]


def test_from_mandate_refuses_to_invent_a_floor():
    with pytest.raises(ValueError, match="min_open_interest"):
        ObservationVerifier.from_mandate({"universe": {}}, StubFetcher(HONEST_CHAIN))


def test_the_floor_has_no_default():
    """Constructing without a floor is an error, not a silent 500."""
    with pytest.raises(TypeError):
        ObservationVerifier(StubFetcher(HONEST_CHAIN))


def test_the_mandate_floor_is_what_gets_enforced():
    import pathlib

    import yaml

    mandate = yaml.safe_load(
        (pathlib.Path(__file__).resolve().parent.parent / "config" / "mandate.yaml").read_text()
    )
    mandate["universe"]["min_open_interest"] = 3000  # above the chain's 2804
    result = ObservationVerifier.from_mandate(
        mandate, StubFetcher(HONEST_CHAIN), today=lambda: TODAY
    ).verify(proposal_with())

    assert not result.verified
    assert result.discrepancies[0].field == "open_interest"
    assert "3000" in result.discrepancies[0].detail
