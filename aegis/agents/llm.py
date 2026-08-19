"""Provider-agnostic access to a language model.

The agents need exactly one thing from an LLM: given a system instruction and a
user message, return text. That is the whole protocol. Keeping it that narrow
is what makes the provider a configuration choice rather than a rewrite -- and
it keeps the boundary honest, since anything an agent cannot express through
:meth:`LLMClient.complete` is something it cannot ask a model to decide.

Backends:

* :class:`AnthropicClient` -- the Anthropic SDK.
* :class:`GeminiClient` -- the ``google-genai`` SDK, defaulting to
  ``gemini-2.5-flash``. When the caller asks for JSON it uses the model's
  native JSON mode rather than hoping the prose parses.
* :class:`NullClient` -- returns None, which every caller already treats as
  "no model available" and degrades around.

:func:`build_llm_client` picks one from the environment. All backends retry
rate limits with exponential backoff: Gemini's free tier allows 10 requests per
minute, and a run across the mandate's eight symbols bursts past that.
"""

from __future__ import annotations

import logging
import os
from time import sleep as _sleep
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

log = logging.getLogger(__name__)

DEFAULT_MAX_ATTEMPTS = 4
#: First backoff step. Doubles per attempt: 4s, 8s, 16s covers a 10 RPM window.
DEFAULT_BACKOFF_SECONDS = 4.0

_no_provider_logged = False


@runtime_checkable
class LLMClient(Protocol):
    """Anything that can turn a system instruction plus a message into text."""

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str | None:
        """Return the model's text, or None if no model is available.

        Args:
            system: The system instruction.
            user: The message to respond to.
            json_mode: Ask for a bare JSON object. Backends with native JSON
                support use it; the prompt asks for JSON regardless, so a
                backend without it still works.
        """
        ...


# --------------------------------------------------------------------------
# retry
# --------------------------------------------------------------------------


def _retry_after(exc: BaseException) -> float | None:
    """Honour a server-supplied Retry-After header when there is one."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


def _with_retries(
    call: Callable[[], Any],
    *,
    is_rate_limited: Callable[[BaseException], bool],
    provider: str,
    max_attempts: int,
    backoff_seconds: float,
    sleeper: Callable[[float], None],
) -> Any:
    """Call ``call``, retrying rate limits only. Everything else propagates."""
    delay = backoff_seconds
    for attempt in range(1, max_attempts + 1):
        try:
            return call()
        except Exception as exc:
            if attempt == max_attempts or not is_rate_limited(exc):
                raise
            wait = _retry_after(exc) or delay
            log.warning(
                "%s rate limited (attempt %d/%d); retrying in %.0fs",
                provider, attempt, max_attempts, wait,
            )
            sleeper(wait)
            delay *= 2
    raise AssertionError("unreachable")  # pragma: no cover


class _Backend:
    """Shared retry configuration."""

    def __init__(
        self,
        *,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self._max_attempts = max(1, int(max_attempts))
        self._backoff = float(backoff_seconds)
        self._sleeper = sleeper or _sleep

    def _call(self, fn, *, is_rate_limited, provider):
        return _with_retries(
            fn,
            is_rate_limited=is_rate_limited,
            provider=provider,
            max_attempts=self._max_attempts,
            backoff_seconds=self._backoff,
            sleeper=self._sleeper,
        )


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------


class AnthropicClient(_Backend):
    """The Anthropic SDK behind the shared protocol."""

    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 4096,
        thinking: bool = False,
        sdk: Any | None = None,
        **retry: Any,
    ) -> None:
        super().__init__(**retry)
        self._model = model
        self._max_tokens = max_tokens
        self._thinking = thinking
        if sdk is not None:
            self._sdk = sdk
        else:
            import anthropic

            self._sdk = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str | None:
        # No native JSON mode: structured outputs are not available on this
        # model, so the caller's prompt does the shaping and parses defensively.
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self._thinking:
            request["thinking"] = {"type": "adaptive"}

        response = self._call(
            lambda: self._sdk.messages.create(**request),
            is_rate_limited=_anthropic_rate_limited,
            provider="anthropic",
        )
        return _anthropic_text(response)

    @property
    def model(self) -> str:
        return self._model


def _anthropic_rate_limited(exc: BaseException) -> bool:
    if getattr(exc, "status_code", None) == 429:
        return True
    return type(exc).__name__ == "RateLimitError"


def _anthropic_text(response: Any) -> str | None:
    """Concatenate text blocks, skipping thinking and other block types."""
    blocks = getattr(response, "content", None) or []
    text = "".join(
        getattr(block, "text", "") for block in blocks if getattr(block, "type", None) == "text"
    )
    return text or None


class GeminiClient(_Backend):
    """The google-genai SDK behind the shared protocol."""

    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = DEFAULT_MODEL,
        max_output_tokens: int | None = None,
        sdk: Any | None = None,
        **retry: Any,
    ) -> None:
        super().__init__(**retry)
        self._model = model
        self._max_output_tokens = max_output_tokens
        if sdk is not None:
            self._sdk = sdk
        else:
            from google import genai

            self._sdk = genai.Client(api_key=api_key) if api_key else genai.Client()

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str | None:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            # Native JSON mode: the model is constrained to emit a bare JSON
            # value, so there is no prose or code fence to strip.
            response_mime_type="application/json" if json_mode else "text/plain",
            max_output_tokens=self._max_output_tokens,
        )
        response = self._call(
            lambda: self._sdk.models.generate_content(
                model=self._model, contents=user, config=config
            ),
            is_rate_limited=_gemini_rate_limited,
            provider="gemini",
        )
        text = getattr(response, "text", None)
        return text or None

    @property
    def model(self) -> str:
        return self._model


def _gemini_rate_limited(exc: BaseException) -> bool:
    return getattr(exc, "code", None) == 429 or getattr(exc, "status", None) == "RESOURCE_EXHAUSTED"


class NullClient:
    """No model configured. Returns None so callers take their degraded path."""

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str | None:
        return None

    @property
    def model(self) -> str:
        return "none"


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def build_llm_client(env: Mapping[str, str] | None = None, **kwargs: Any) -> LLMClient | None:
    """Pick a backend from the environment.

    ``ANTHROPIC_API_KEY`` wins, then ``GEMINI_API_KEY``. With neither set this
    returns None -- the agents degrade rather than fail, so a missing key costs
    the written explanation and never the trade.
    """
    global _no_provider_logged
    env = os.environ if env is None else env

    if env.get("ANTHROPIC_API_KEY"):
        return AnthropicClient(env["ANTHROPIC_API_KEY"], **kwargs)
    if env.get("GEMINI_API_KEY"):
        return GeminiClient(env["GEMINI_API_KEY"], **kwargs)

    if not _no_provider_logged:
        log.info(
            "no LLM provider configured (set ANTHROPIC_API_KEY or GEMINI_API_KEY); "
            "proposals will carry no thesis and the critic will fail closed"
        )
        _no_provider_logged = True
    return None
