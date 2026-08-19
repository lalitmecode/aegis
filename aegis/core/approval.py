"""Human approval tokens.

An :class:`ApprovalToken` is consent to execute *one specific proposal, as it
read at the moment a human looked at it, for the next two minutes*. It is bound
to both the proposal id and a hash of the proposal's contents, so it cannot be
replayed against a different proposal or against an edited version of the same
one. It expires quickly because consent to trade is priced: the market moves.

This module deliberately raises nothing and knows nothing about the gateway --
:meth:`ApprovalToken.verify` returns a verdict, and the caller decides what a
failure means.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from aegis.core.proposal import TradeProposal

#: Mandate default: consent goes stale after two minutes.
APPROVAL_TTL_SECONDS = 120


def _as_utc(moment: datetime) -> datetime:
    """Treat naive datetimes as UTC so comparisons never raise."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class TokenVerdict:
    """Outcome of verifying a token against a proposal."""

    ok: bool
    code: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ApprovalToken:
    """Time-boxed human consent bound to exact proposal contents."""

    proposal_id: str
    proposal_hash: str
    issued_at: datetime
    expires_at: datetime
    operator: str

    @classmethod
    def issue(
        cls,
        proposal: TradeProposal,
        operator: str,
        *,
        ttl_seconds: int = APPROVAL_TTL_SECONDS,
        now: datetime | None = None,
    ) -> "ApprovalToken":
        """Mint a token for ``proposal`` as it currently reads."""
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be positive, got {ttl_seconds}")
        issued = _as_utc(now or _now())
        return cls(
            proposal_id=proposal.proposal_id,
            proposal_hash=proposal.content_hash(),
            issued_at=issued,
            expires_at=issued + timedelta(seconds=ttl_seconds),
            operator=operator,
        )

    def verify(self, proposal: TradeProposal, *, now: datetime | None = None) -> TokenVerdict:
        """Check this token authorises ``proposal`` right now.

        Checks, in order: the token names this proposal, the proposal still
        hashes to what was approved, and the clock is inside the validity
        window.
        """
        if self.proposal_id != proposal.proposal_id:
            return TokenVerdict(
                False,
                "proposal_mismatch",
                f"token authorises proposal {self.proposal_id!r}, "
                f"not {proposal.proposal_id!r}",
            )

        current_hash = proposal.content_hash()
        if self.proposal_hash != current_hash:
            return TokenVerdict(
                False,
                "content_mismatch",
                f"proposal {proposal.proposal_id!r} changed after approval "
                f"(approved {self.proposal_hash[:12]}..., now {current_hash[:12]}...)",
            )

        moment = _as_utc(now or _now())
        if moment >= _as_utc(self.expires_at):
            age = (moment - _as_utc(self.issued_at)).total_seconds()
            return TokenVerdict(
                False,
                "expired",
                f"approval expired {age:.0f}s after issue "
                f"(expires_at {self.expires_at.isoformat()})",
            )
        if moment < _as_utc(self.issued_at):
            return TokenVerdict(
                False,
                "not_yet_valid",
                f"token issued in the future ({self.issued_at.isoformat()})",
            )

        return TokenVerdict(True)

    @property
    def ttl_seconds(self) -> float:
        return (_as_utc(self.expires_at) - _as_utc(self.issued_at)).total_seconds()
