"""Provider-agnostic access to a language model.

The agents need exactly one thing from an LLM: given a system instruction and a
user message, return text. That is the whole protocol. Keeping it that narrow
is what makes the provider a configuration choice rather than a rewrite -- and
it keeps the boundary honest, since anything an agent cannot express through
:meth:`LLMClient.complete` is something it cannot ask a model to decide.

Backends:

* :class:`AnthropicClient` -- the Anthropic SDK.
* :class:`GeminiClient` -- the ``google-genai`` SDK, defaulting to
  ``gemini-3.6-flash``. When the caller asks for JSON it uses the model's
  native JSON mode rather than hoping the prose parses.

Model ids are the part of this that rots: providers deprecate them on their own
schedule, and a 404 mid-demo is not the moment to edit source. Both backends
read their model from the environment (``GEMINI_MODEL``, ``ANTHROPIC_MODEL``),
so moving to a successor model is a config change.
* :class:`NullClient` -- returns None, which every caller already treats as
  "no model available" and degrades around.

:func:`build_llm_client` picks one from the environment. All backends retry transient
failures with exponential backoff -- 429 rate limits (Gemini's free tier allows
10 requests per minute, and a run across the mandate's eight symbols bursts past
that) and 5xx server errors, which the provider itself describes as temporary.
A 4xx that is not 429 is a decision, not a blip, and propagates immediately.
"""

from __future__ import annotations

import logging
import os
from time import sleep as _sleep
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

log = logging.getLogger(__name__)

#: Model ids, overridable per the note above. ``gemini-2.5-flash`` is gone:
#: Google returns 404 for it on newly issued API keys and points at 3.6.
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"

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

    @property
    def model(self) -> str:
        """The model id actually in use, after any environment override."""
        ...

    @property
    def name(self) -> str:
        """Human-readable label for whatever produced the text."""
        ...


# --------------------------------------------------------------------------
# retry
# --------------------------------------------------------------------------


def _display_name(model: str) -> str:
    """Turn a model id into a label: ``claude-sonnet-4-6`` -> ``Claude Sonnet 4.6``.

    Derived rather than hardcoded. A constant per class would go stale the
    moment ``GEMINI_MODEL`` or ``ANTHROPIC_MODEL`` points somewhere else --
    reintroducing exactly the drift this label exists to prevent.
    """
    parts: list[str] = []
    for segment in model.split("-"):
        numeric = segment.replace(".", "").isdigit()
        if numeric and parts and parts[-1].replace(".", "").isdigit():
            # Anthropic spells versions with dashes: 4-6 is 4.6.
            parts[-1] = f"{parts[-1]}.{segment}"
        elif numeric:
            parts.append(segment)
        else:
            parts.append(segment.capitalize())
    return " ".join(parts) or model


def _model_from_env(variable: str, default: str) -> str:
    """Resolve a model id from the process environment, falling back to ours."""
    return os.environ.get(variable) or default


def _retry_after(exc: BaseException) -> float | None:
    """Honour a server-supplied Retry-After header when there is one."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


def _status_of(exc: BaseException) -> int | None:
    """HTTP status carried by a provider exception, if any."""
    for attribute in ("code", "status_code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value
    return None


def _is_transient_status(status: int | None) -> bool:
    """429 is a rate limit; 5xx is the provider having a bad moment."""
    return status is not None and (status == 429 or 500 <= status < 600)


def _with_retries(
    call: Callable[[], Any],
    *,
    is_retryable: Callable[[BaseException], bool],
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
            if attempt == max_attempts or not is_retryable(exc):
                raise
            wait = _retry_after(exc) or delay
            log.warning(
                "%s call failed with %s (attempt %d/%d); retrying in %.0fs",
                provider, _status_of(exc) or type(exc).__name__, attempt, max_attempts, wait,
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

    def _call(self, fn, *, is_retryable, provider):
        return _with_retries(
            fn,
            is_retryable=is_retryable,
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

    #: Env var that overrides the model id.
    MODEL_ENV = "ANTHROPIC_MODEL"
    DEFAULT_MODEL = DEFAULT_ANTHROPIC_MODEL

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str | None = None,
        max_tokens: int = 4096,
        thinking: bool = False,
        sdk: Any | None = None,
        **retry: Any,
    ) -> None:
        super().__init__(**retry)
        self._model = model or _model_from_env(self.MODEL_ENV, DEFAULT_ANTHROPIC_MODEL)
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
            is_retryable=_anthropic_retryable,
            provider="anthropic",
        )
        return _anthropic_text(response)

    @property
    def model(self) -> str:
        return self._model

    @property
    def name(self) -> str:
        return _display_name(self._model)


def _anthropic_retryable(exc: BaseException) -> bool:
    if _is_transient_status(_status_of(exc)):
        return True
    return type(exc).__name__ in {"RateLimitError", "InternalServerError", "APIConnectionError"}


def _anthropic_text(response: Any) -> str | None:
    """Concatenate text blocks, skipping thinking and other block types."""
    blocks = getattr(response, "content", None) or []
    text = "".join(
        getattr(block, "text", "") for block in blocks if getattr(block, "type", None) == "text"
    )
    return text or None


class GeminiClient(_Backend):
    """The google-genai SDK behind the shared protocol."""

    #: Env var that overrides the model id.
    MODEL_ENV = "GEMINI_MODEL"
    DEFAULT_MODEL = DEFAULT_GEMINI_MODEL

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str | None = None,
        max_output_tokens: int | None = None,
        sdk: Any | None = None,
        **retry: Any,
    ) -> None:
        super().__init__(**retry)
        self._model = model or _model_from_env(self.MODEL_ENV, DEFAULT_GEMINI_MODEL)
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
            is_retryable=_gemini_retryable,
            provider="gemini",
        )
        text = getattr(response, "text", None)
        return text or None

    @property
    def model(self) -> str:
        return self._model

    @property
    def name(self) -> str:
        return _display_name(self._model)


def _gemini_retryable(exc: BaseException) -> bool:
    if _is_transient_status(_status_of(exc)):
        return True
    # google-genai spells the reason as well as the code.
    return getattr(exc, "status", None) in {"RESOURCE_EXHAUSTED", "UNAVAILABLE"}


class NullClient:
    """No model configured. Returns None so callers take their degraded path."""

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str | None:
        return None

    @property
    def model(self) -> str:
        return "none"

    @property
    def name(self) -> str:
        return "no model"


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def build_llm_client(env: Mapping[str, str] | None = None, **kwargs: Any) -> LLMClient | None:
    """Pick a backend from the environment.

    ``ANTHROPIC_API_KEY`` wins, then ``GEMINI_API_KEY``. With neither set this
    returns None -- the agents degrade rather than fail, so a missing key costs
    the written explanation and never the trade.

    ``GEMINI_MODEL`` / ``ANTHROPIC_MODEL`` in ``env`` override the default model
    id; an explicit ``model=`` keyword still wins over both.
    """
    global _no_provider_logged
    env = os.environ if env is None else env

    if env.get("ANTHROPIC_API_KEY"):
        kwargs.setdefault("model", env.get(AnthropicClient.MODEL_ENV))
        return AnthropicClient(env["ANTHROPIC_API_KEY"], **kwargs)
    if env.get("GEMINI_API_KEY"):
        kwargs.setdefault("model", env.get(GeminiClient.MODEL_ENV))
        return GeminiClient(env["GEMINI_API_KEY"], **kwargs)

    if not _no_provider_logged:
        log.info(
            "no LLM provider configured (set ANTHROPIC_API_KEY or GEMINI_API_KEY); "
            "proposals will carry no thesis and the critic will fail closed"
        )
        _no_provider_logged = True
    return None
