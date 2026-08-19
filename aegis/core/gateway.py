"""The only code path permitted to submit orders to the broker.

Every gate the mandate promises is enforced here, at the boundary, because
this is the last place where refusing is still cheap. The gateway trusts
nothing it is handed: not a risk decision computed upstream, not a portfolio
snapshot, not the claim that a human said yes.

Order of enforcement in :meth:`ExecutionGateway.submit`:

1. the proposal must sit inside the mandate's declared universe and structures
   (free, so it runs first and spares the network a pointless call);
2. the observation verifier reconciles the agent's claims against live market
   data -- claims it made about its own legs are not taken on trust;
3. the risk guard is re-run against a **freshly fetched** portfolio state;
4. an approval token must be present, unexpired, and bound to this exact
   proposal's contents;
5. only then is the multi-leg order built and sent.

Any failure is written to the audit chain and then raised as a
:class:`GovernanceError`. Nothing fails quietly, and nothing returns a
sentinel a caller could ignore.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from time import sleep as _sleep
from typing import Any, Callable, Mapping, Protocol

from alpaca.trading.enums import OrderClass
from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest

from aegis.core.approval import ApprovalToken
from aegis.core.proposal import PortfolioState, RiskDecision, TradeProposal
from aegis.core.retry import (
    DEFAULT_BACKOFF_SECONDS,
    DEFAULT_MAX_ATTEMPTS,
    TRANSIENT_ERRORS,
)


class GovernanceError(RuntimeError):
    """A submission was refused. Never raised for a merely unlucky trade."""


class ApprovalMissing(GovernanceError):
    """No human approval token was presented."""


class ApprovalExpired(GovernanceError):
    """The approval token is past its expiry."""


class ApprovalMismatch(GovernanceError):
    """The token does not authorise this proposal, as it now reads."""


class MandateViolation(GovernanceError):
    """The proposal falls outside the mandate's universe or structures."""


class VerificationFailed(GovernanceError):
    """Observed market data contradicts the proposal, or could not be obtained."""


class RiskRejected(GovernanceError):
    """The risk guard refused the proposal on a freshly fetched state."""


class BrokerSubmissionError(GovernanceError):
    """The broker could not be reached, or refused the order."""


_VERDICT_ERRORS: Mapping[str, type[GovernanceError]] = {
    "proposal_mismatch": ApprovalMismatch,
    "content_mismatch": ApprovalMismatch,
    "expired": ApprovalExpired,
    "not_yet_valid": ApprovalMismatch,
}


class RiskGuard(Protocol):
    """Deterministic numeric enforcement of ``risk_limits``."""

    def evaluate(self, proposal: TradeProposal, state: PortfolioState) -> RiskDecision: ...


class AuditChain(Protocol):
    """Append-only decision log."""

    def record(self, event: str, payload: Mapping[str, Any]) -> Any: ...


class ObservationVerifier(Protocol):
    """Reconciles agent claims against observed market data."""

    def verify(self, proposal: TradeProposal) -> Any: ...


class PortfolioSource(Protocol):
    """Something that can produce a *current* portfolio snapshot.

    The gateway takes a source rather than a snapshot so that freshness is
    structural: there is no way to hand it a stale state.
    """

    def fetch(self) -> PortfolioState: ...


@dataclass(frozen=True, slots=True)
class OrderResult:
    """Receipt for an order that actually reached the broker."""

    proposal_id: str
    proposal_hash: str
    broker_order_id: str
    status: str
    operator: str
    submitted_at: datetime
    client_order_id: str


class ExecutionGateway:
    """Submits orders, but only after every gate has been re-checked here."""

    def __init__(
        self,
        mandate: Mapping[str, Any],
        guard: RiskGuard,
        audit: AuditChain,
        trading_client: Any,
        *,
        verifier: ObservationVerifier | None = None,
        clock: Callable[[], datetime] | None = None,
        sleeper: Callable[[float], None] | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    ) -> None:
        self._mandate = mandate
        self._guard = guard
        self._audit = audit
        self._trading = trading_client
        self._verifier = verifier
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._sleep = sleeper or _sleep
        self._max_attempts = max(1, int(max_attempts))
        self._backoff = float(backoff_seconds)

    def submit(
        self,
        proposal: TradeProposal,
        state: PortfolioSource,
        human_token: ApprovalToken | None,
    ) -> OrderResult:
        """Run every gate, then submit. Raises :class:`GovernanceError` on any refusal.

        Args:
            proposal: The structure to open.
            state: A source of *current* portfolio state. The guard is re-run
                against ``state.fetch()``, never against anything cached.
            human_token: Approval bound to this proposal's contents.
        """
        now = self._clock()

        # Read only as an audit label: unverified until the approval gate below.
        operator = human_token.operator if human_token is not None else None

        self._check_mandate(proposal, operator=operator)
        self._verify_observations(proposal, operator=operator)

        # Requirement: re-evaluate. Any decision computed upstream is ignored.
        fresh_state = state.fetch()
        decision = self._guard.evaluate(proposal, fresh_state)
        if not decision.approved:
            self._refuse(
                RiskRejected,
                proposal,
                "; ".join(decision.reasons) or "risk guard refused the proposal",
                operator=operator,
            )

        if human_token is None:
            self._refuse(
                ApprovalMissing,
                proposal,
                "no approval token presented",
                operator=None,
            )

        verdict = human_token.verify(proposal, now=now)
        if not verdict.ok:
            self._refuse(
                _VERDICT_ERRORS.get(verdict.code or "", ApprovalMismatch),
                proposal,
                verdict.reason or f"approval rejected ({verdict.code})",
                operator=human_token.operator,
                code=verdict.code,
            )

        client_order_id = self._client_order_id(proposal)
        order = self._submit_with_retry(
            self._build_order(proposal, client_order_id),
            proposal,
            operator=human_token.operator,
        )

        result = OrderResult(
            proposal_id=proposal.proposal_id,
            proposal_hash=proposal.content_hash(),
            broker_order_id=str(getattr(order, "id", "")),
            status=str(getattr(getattr(order, "status", ""), "value", getattr(order, "status", ""))),
            operator=human_token.operator,
            submitted_at=self._clock(),
            client_order_id=client_order_id,
        )
        self._audit.record(
            "SUBMITTED",
            {
                "proposal_id": result.proposal_id,
                "proposal_hash": result.proposal_hash,
                "broker_order_id": result.broker_order_id,
                "client_order_id": result.client_order_id,
                "status": result.status,
                "operator": result.operator,
                "submitted_at": result.submitted_at.isoformat(),
            },
        )
        return result

    def _refuse(
        self,
        error: type[GovernanceError],
        proposal: TradeProposal,
        reason: str,
        *,
        operator: str | None,
        code: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        """Audit the refused attempt, then raise. Never returns."""
        payload: dict[str, Any] = {
            "proposal_id": proposal.proposal_id,
            "proposal_hash": proposal.content_hash(),
            "error": error.__name__,
            "code": code,
            "reason": reason,
            "operator": operator,
            "at": self._clock().isoformat(),
        }
        if extra:
            payload.update(extra)
        self._audit.record("REJECTED", payload)
        raise error(f"{proposal.proposal_id}: {reason}")

    def _verify_observations(self, proposal: TradeProposal, *, operator: str | None) -> None:
        """Refuse anything whose claims the market does not corroborate.

        Discrepancies are never silently corrected -- an agent that misreports
        its own inputs has failed, and patching the numbers here would hide
        that failure from the audit trail.
        """
        if self._verifier is None:
            self._refuse(
                VerificationFailed,
                proposal,
                "no observation verifier configured; refusing to trade unverified claims",
                operator=operator,
            )

        try:
            result = self._verifier.verify(proposal)
        except Exception as exc:
            self._refuse(
                VerificationFailed,
                proposal,
                f"verification could not complete: {type(exc).__name__}: {exc}",
                operator=operator,
            )

        if not result.verified:
            payload = getattr(result, "as_audit_payload", lambda: [])()
            self._refuse(
                VerificationFailed,
                proposal,
                f"observed market data contradicts the proposal: {result.summary()}",
                operator=operator,
                extra={"discrepancies": payload},
            )

    def _check_mandate(self, proposal: TradeProposal, *, operator: str) -> None:
        """Last-line check of the mandate's universe and structure whitelists.

        Fails closed: a mandate missing these keys refuses everything rather
        than defaulting to permissive.
        """
        universe = self._mandate.get("universe") or {}
        strategy = self._mandate.get("strategy") or {}
        allowed_symbols = universe.get("allowed_symbols") or []
        permitted = strategy.get("permitted_structures") or []
        prohibited = strategy.get("prohibited_structures") or []

        if proposal.underlying not in allowed_symbols:
            self._refuse(
                MandateViolation,
                proposal,
                f"{proposal.underlying} is not in the mandate universe",
                operator=operator,
            )
        if proposal.structure in prohibited or proposal.structure not in permitted:
            self._refuse(
                MandateViolation,
                proposal,
                f"structure {proposal.structure!r} is not a permitted structure",
                operator=operator,
            )

    def _client_order_id(self, proposal: TradeProposal) -> str:
        """Deterministic id so a retried submit cannot double-fill.

        If a submit succeeds but the response is lost, the retry carries the
        same client_order_id and the broker rejects it as a duplicate rather
        than opening a second position.
        """
        return f"aegis-{proposal.proposal_id}-{proposal.content_hash()[:12]}"

    def _build_order(self, proposal: TradeProposal, client_order_id: str) -> LimitOrderRequest:
        legs = [
            OptionLegRequest(
                symbol=leg.symbol,
                ratio_qty=leg.ratio_qty,
                side=leg.side,
                position_intent=leg.position_intent,
            )
            for leg in proposal.legs
        ]
        return LimitOrderRequest(
            qty=proposal.quantity,
            limit_price=float(proposal.limit_price),
            order_class=OrderClass.MLEG,
            time_in_force=proposal.time_in_force,
            legs=legs,
            client_order_id=client_order_id,
        )

    def _submit_with_retry(
        self,
        order: LimitOrderRequest,
        proposal: TradeProposal,
        *,
        operator: str,
    ) -> Any:
        """Submit, retrying transient connection failures with exponential backoff."""
        delay = self._backoff
        last_error: BaseException | None = None

        for attempt in range(1, self._max_attempts + 1):
            try:
                return self._trading.submit_order(order)
            except TRANSIENT_ERRORS as exc:
                last_error = exc
                self._audit.record(
                    "BROKER_RETRY",
                    {
                        "proposal_id": proposal.proposal_id,
                        "attempt": attempt,
                        "of": self._max_attempts,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                if attempt < self._max_attempts:
                    self._sleep(delay)
                    delay *= 2
            except Exception as exc:  # broker refused the order outright
                last_error = exc
                self._refuse(
                    BrokerSubmissionError,
                    proposal,
                    f"broker rejected the order: {type(exc).__name__}: {exc}",
                    operator=operator,
                )

        self._refuse(
            BrokerSubmissionError,
            proposal,
            f"broker unreachable after {self._max_attempts} attempts: "
            f"{type(last_error).__name__}: {last_error}",
            operator=operator,
        )
        raise AssertionError("unreachable")  # pragma: no cover
