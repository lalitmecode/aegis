"""The agent that argues against a proposal.

The critic reads the mandate and a proposal's thesis and tries to find fault
with it -- tension with a clause, a risk the thesis waves past, an assumption
that does not survive contact with the numbers.

Two constraints define what it is:

**It can only object.** The model is never asked for a verdict. Its response
schema has room for concerns and clause references and nothing else; ``passed``
is computed by :meth:`CriticAgent.review` as "raised no concerns". A model that
wants to enthusiastically approve a trade has no field to write that into. The
strongest thing it can do is stay silent, and silence is not permission.

**Its verdict binds nothing.** A :class:`Critique` is advisory input for the
human reading the proposal. :class:`~aegis.core.gateway.ExecutionGateway` does
not accept one and has no code path that consults it, so a clean critique can
never override a risk-guard refusal -- the guard runs inside ``submit()``
against a freshly fetched state regardless of what the critic said.

Failure is silence-shaped, so it fails closed: no client, an API error, or an
unparseable response all yield ``passed=False`` with the reason as a concern.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Mapping

DEFAULT_MODEL = "claude-sonnet-4-6"

log = logging.getLogger(__name__)

_CRITIC_SYSTEM = """You are the critic for Aegis, a mandate-governed options \
trading agent. You are reviewing a trade another agent has proposed, before a \
human decides whether to approve it.

Your job is to challenge it. Look for tension with the mandate, risks the thesis \
understates or ignores, and claims the stated numbers do not support. Reference \
specific mandate clauses by their path, for example \
"strategy.delta_limits.short_leg_abs_delta_max" or "risk_limits.max_open_positions".

You cannot approve this trade. You have no mechanism to do so, and raising no \
concerns is not an endorsement -- separate deterministic checks decide whether \
the trade is permitted, and they run whatever you say. Do not comment on whether \
the trade should proceed.

Report only concerns you can support from the mandate and the figures given. Do \
not invent numbers, and do not manufacture an objection when you have none.

Respond with a single JSON object and nothing else:

{"concerns": ["..."], "clause_refs": ["..."]}

Both lists may be empty. Do not wrap the JSON in code fences or add commentary."""


@dataclass(frozen=True, slots=True)
class Critique:
    """The critic's advisory reading of a proposal.

    ``passed`` is derived, never asserted by the model: it is True only when
    ``concerns`` is empty. It means "the critic found nothing to say", not
    "this trade is approved".
    """

    passed: bool
    concerns: tuple[str, ...] = ()
    clause_refs: tuple[str, ...] = ()
    raw: str | None = None

    def summary(self) -> str:
        if self.passed:
            return "no concerns raised"
        return "; ".join(self.concerns)


class CriticAgent:
    """Sends a proposal and the mandate to Claude and collects objections."""

    def __init__(
        self,
        llm_client: Any | None = None,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 4096,
    ) -> None:
        self._llm = llm_client
        self._model = model
        self._max_tokens = max_tokens

    def review(self, proposal, mandate: Mapping[str, Any] | str) -> Critique:
        """Review ``proposal`` against ``mandate``. Never raises."""
        if self._llm is None:
            return Critique(
                False,
                ("no LLM client configured; the proposal has not been challenged",),
            )

        try:
            response = self._llm.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                thinking={"type": "adaptive"},
                output_config={"effort": "medium"},
                system=_CRITIC_SYSTEM,
                messages=[{"role": "user", "content": self._prompt(proposal, mandate)}],
            )
        except Exception as exc:
            log.warning("critic call failed: %s", exc)
            return Critique(False, (f"critic could not run: {type(exc).__name__}: {exc}",))

        raw = _text_of(response).strip()
        parsed = _parse_json_object(raw)
        if parsed is None:
            log.warning("critic returned unparseable output")
            return Critique(
                False,
                ("critic response could not be parsed; treating as unreviewed",),
                raw=raw,
            )

        concerns = tuple(str(c) for c in parsed.get("concerns") or [] if str(c).strip())
        clause_refs = tuple(str(r) for r in parsed.get("clause_refs") or [] if str(r).strip())

        # Derived, not claimed: the model has no verdict field to write into.
        return Critique(
            passed=not concerns,
            concerns=concerns,
            clause_refs=clause_refs,
            raw=raw,
        )

    def _prompt(self, proposal, mandate: Mapping[str, Any] | str) -> str:
        legs = "\n".join(
            f"  {leg.side.value if hasattr(leg.side, 'value') else leg.side} "
            f"{leg.symbol} (delta {leg.delta})"
            for leg in proposal.legs
        )
        return (
            f"MANDATE\n{_mandate_text(mandate)}\n\n"
            f"PROPOSAL\n"
            f"  id: {proposal.proposal_id}\n"
            f"  underlying: {proposal.underlying}\n"
            f"  structure: {proposal.structure}\n"
            f"  quantity: {proposal.quantity}\n"
            f"  net credit: {proposal.limit_price}\n"
            f"  stated max loss: ${proposal.max_loss_usd}\n"
            f"  legs:\n{legs}\n\n"
            f"THESIS\n{proposal.thesis or '(none provided)'}\n"
        )


def _mandate_text(mandate: Mapping[str, Any] | str) -> str:
    if isinstance(mandate, str):
        return mandate
    import yaml

    return yaml.safe_dump(dict(mandate), sort_keys=False)


def _text_of(response: Any) -> str:
    """Concatenate text blocks, skipping thinking blocks."""
    blocks = getattr(response, "content", None) or []
    return "".join(
        getattr(block, "text", "") for block in blocks if getattr(block, "type", None) == "text"
    )


def _parse_json_object(raw: str) -> dict | None:
    """Pull the first JSON object out of a response.

    Structured outputs are not available on this model, so the response is
    prose-shaped by instruction rather than by schema and has to be parsed
    defensively.
    """
    if not raw:
        return None
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
