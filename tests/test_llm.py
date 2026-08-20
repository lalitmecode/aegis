"""Provider-layer tests. Every SDK is injected as a stub; nothing calls a real API."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors

from aegis.agents import llm as llm_module
from aegis.agents.llm import (
    AnthropicClient,
    GeminiClient,
    LLMClient,
    NullClient,
    build_llm_client,
)


@pytest.fixture(autouse=True)
def _clean_model_env(monkeypatch):
    """Model overrides must not leak in from the developer's shell."""
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)


def rate_limited() -> Exception:
    return genai_errors.ClientError(429, {"error": {"message": "quota exceeded"}})


class StubGeminiSDK:
    """Mimics google-genai's `client.models.generate_content(...)`."""

    def __init__(self, text: str = '{"concerns": [], "clause_refs": []}', errors=()):
        self.text = text
        self._errors = list(errors)
        self.calls: list[dict] = []
        self.models = self

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self._errors:
            raise self._errors.pop(0)
        return SimpleNamespace(text=self.text)


class StubAnthropicSDK:
    """Mimics anthropic's `client.messages.create(...)`."""

    def __init__(self, text: str = "A short thesis.", errors=()):
        self.text = text
        self._errors = list(errors)
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._errors:
            raise self._errors.pop(0)
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="hidden"),
                SimpleNamespace(type="text", text=self.text),
            ]
        )


def gemini(sdk=None, **kw) -> GeminiClient:
    kw.setdefault("sleeper", lambda _s: None)
    return GeminiClient(sdk=sdk or StubGeminiSDK(), **kw)


def anthropic(sdk=None, **kw) -> AnthropicClient:
    kw.setdefault("sleeper", lambda _s: None)
    return AnthropicClient(sdk=sdk or StubAnthropicSDK(), **kw)


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def test_anthropic_key_selects_anthropic():
    client = build_llm_client({"ANTHROPIC_API_KEY": "sk-test"}, sdk=StubAnthropicSDK())
    assert isinstance(client, AnthropicClient)
    assert client.model == "claude-sonnet-4-6"


def test_gemini_key_selects_gemini():
    client = build_llm_client({"GEMINI_API_KEY": "g-test"}, sdk=StubGeminiSDK())
    assert isinstance(client, GeminiClient)
    assert client.model == "gemini-3.6-flash"


def test_anthropic_wins_when_both_are_set():
    client = build_llm_client(
        {"ANTHROPIC_API_KEY": "sk-test", "GEMINI_API_KEY": "g-test"}, sdk=StubAnthropicSDK()
    )
    assert isinstance(client, AnthropicClient)


def test_no_key_returns_none():
    assert build_llm_client({}) is None


def test_the_missing_provider_notice_is_logged_once(caplog):
    llm_module._no_provider_logged = False
    with caplog.at_level(logging.INFO, logger="aegis.agents.llm"):
        build_llm_client({})
        build_llm_client({})
        build_llm_client({})
    notices = [r for r in caplog.records if "no LLM provider configured" in r.message]
    assert len(notices) == 1


def test_every_backend_satisfies_the_protocol():
    for client in (NullClient(), gemini(), anthropic()):
        assert isinstance(client, LLMClient)


def test_null_client_returns_none():
    assert NullClient().complete("system", "user") is None
    assert NullClient().complete("system", "user", json_mode=True) is None


# --------------------------------------------------------------------------
# model ids are configuration, not code
# --------------------------------------------------------------------------


def test_the_gemini_default_is_a_model_google_still_serves():
    """gemini-2.5-flash 404s on newly issued keys; Google points at 3.6."""
    from aegis.agents.llm import DEFAULT_GEMINI_MODEL

    assert DEFAULT_GEMINI_MODEL == "gemini-3.6-flash"
    assert GeminiClient(sdk=StubGeminiSDK()).model == "gemini-3.6-flash"


def test_gemini_model_is_overridable_from_the_environment():
    client = build_llm_client(
        {"GEMINI_API_KEY": "g", "GEMINI_MODEL": "gemini-4.0-pro"}, sdk=StubGeminiSDK()
    )
    assert client.model == "gemini-4.0-pro"


def test_anthropic_model_is_overridable_from_the_environment():
    client = build_llm_client(
        {"ANTHROPIC_API_KEY": "k", "ANTHROPIC_MODEL": "claude-opus-5"}, sdk=StubAnthropicSDK()
    )
    assert client.model == "claude-opus-5"


def test_an_explicit_model_beats_the_environment():
    client = build_llm_client(
        {"GEMINI_API_KEY": "g", "GEMINI_MODEL": "from-env"},
        sdk=StubGeminiSDK(),
        model="explicit",
    )
    assert client.model == "explicit"


def test_an_explicit_model_beats_the_process_environment(monkeypatch):
    """Precedence is explicit > environment > default, on both construction paths."""
    monkeypatch.setenv("GEMINI_MODEL", "from-env")
    monkeypatch.setenv("ANTHROPIC_MODEL", "from-env")
    assert GeminiClient(sdk=StubGeminiSDK(), model="explicit").model == "explicit"
    assert AnthropicClient(sdk=StubAnthropicSDK(), model="explicit").model == "explicit"


def test_the_process_environment_is_honoured_on_direct_construction(monkeypatch):
    """A deprecation should be fixable without going through build_llm_client."""
    monkeypatch.setenv("GEMINI_MODEL", "gemini-4.0-flash")
    assert GeminiClient(sdk=StubGeminiSDK()).model == "gemini-4.0-flash"


def test_the_overridden_model_is_what_reaches_the_api():
    sdk = StubGeminiSDK()
    GeminiClient(sdk=sdk, model="gemini-4.0-pro", sleeper=lambda _s: None).complete("sys", "user")
    assert sdk.calls[0]["model"] == "gemini-4.0-pro"


# --------------------------------------------------------------------------
# gemini
# --------------------------------------------------------------------------


def test_gemini_returns_the_models_text():
    sdk = StubGeminiSDK(text='{"concerns": ["width is thin"], "clause_refs": []}')
    assert gemini(sdk).complete("sys", "user") == '{"concerns": ["width is thin"], "clause_refs": []}'


def test_gemini_uses_native_json_mode_when_asked():
    sdk = StubGeminiSDK()
    gemini(sdk).complete("sys", "user", json_mode=True)
    assert sdk.calls[0]["config"].response_mime_type == "application/json"


def test_gemini_asks_for_plain_text_otherwise():
    sdk = StubGeminiSDK(text="A short thesis.")
    gemini(sdk).complete("sys", "user")
    assert sdk.calls[0]["config"].response_mime_type == "text/plain"


def test_gemini_passes_the_system_instruction_separately():
    sdk = StubGeminiSDK()
    gemini(sdk).complete("you are the critic", "review this")
    call = sdk.calls[0]
    assert call["config"].system_instruction == "you are the critic"
    assert call["contents"] == "review this"
    assert call["model"] == "gemini-3.6-flash"


def test_gemini_json_output_parses():
    """The critic's parser must accept what native JSON mode returns."""
    import json

    sdk = StubGeminiSDK(text='{"concerns": ["a", "b"], "clause_refs": ["risk_limits"]}')
    raw = gemini(sdk).complete("sys", "user", json_mode=True)
    assert json.loads(raw)["concerns"] == ["a", "b"]


def test_gemini_empty_response_is_none():
    assert gemini(StubGeminiSDK(text="")).complete("sys", "user") is None


# --------------------------------------------------------------------------
# anthropic
# --------------------------------------------------------------------------


def test_anthropic_returns_text_blocks_only():
    """Thinking blocks must not leak into the answer."""
    assert anthropic(StubAnthropicSDK(text="Thesis.")).complete("sys", "user") == "Thesis."


def test_anthropic_sends_system_and_user_separately():
    sdk = StubAnthropicSDK()
    anthropic(sdk).complete("you are the analyst", "these are the facts")
    call = sdk.calls[0]
    assert call["system"] == "you are the analyst"
    assert call["messages"] == [{"role": "user", "content": "these are the facts"}]
    assert "thinking" not in call


def test_anthropic_thinking_is_opt_in():
    sdk = StubAnthropicSDK()
    anthropic(sdk, thinking=True).complete("sys", "user")
    assert sdk.calls[0]["thinking"] == {"type": "adaptive"}


# --------------------------------------------------------------------------
# rate limits
# --------------------------------------------------------------------------


def test_a_429_retries_then_succeeds():
    sdk = StubGeminiSDK(text="recovered", errors=[rate_limited(), rate_limited()])
    assert gemini(sdk).complete("sys", "user") == "recovered"
    assert len(sdk.calls) == 3


def test_anthropic_429_retries_then_succeeds():
    class FakeRateLimit(Exception):
        status_code = 429

    sdk = StubAnthropicSDK(text="recovered", errors=[FakeRateLimit()])
    assert anthropic(sdk).complete("sys", "user") == "recovered"
    assert len(sdk.calls) == 2


def test_a_rate_limit_error_is_recognised_by_class_name():
    """The Anthropic SDK's RateLimitError carries no status_code attribute."""

    class RateLimitError(Exception):
        pass

    sdk = StubAnthropicSDK(text="recovered", errors=[RateLimitError()])
    assert anthropic(sdk).complete("sys", "user") == "recovered"


def test_backoff_is_exponential():
    delays: list[float] = []
    sdk = StubGeminiSDK(errors=[rate_limited(), rate_limited(), rate_limited()])
    GeminiClient(sdk=sdk, sleeper=delays.append, backoff_seconds=4.0, max_attempts=4).complete(
        "sys", "user"
    )
    assert delays == [4.0, 8.0, 16.0]


def test_a_retry_after_header_is_honoured():
    error = rate_limited()
    error.response = SimpleNamespace(headers={"retry-after": "1.5"})
    delays: list[float] = []
    sdk = StubGeminiSDK(errors=[error])
    GeminiClient(sdk=sdk, sleeper=delays.append, backoff_seconds=4.0).complete("sys", "user")
    assert delays == [1.5]


def test_persistent_rate_limiting_eventually_raises():
    sdk = StubGeminiSDK(errors=[rate_limited() for _ in range(10)])
    with pytest.raises(genai_errors.ClientError):
        gemini(sdk, max_attempts=3).complete("sys", "user")
    assert len(sdk.calls) == 3


def test_a_non_rate_limit_error_is_not_retried():
    """A bad request is a decision, not a blip."""
    sdk = StubGeminiSDK(errors=[genai_errors.ClientError(400, {"error": {"message": "bad"}})])
    with pytest.raises(genai_errors.ClientError):
        gemini(sdk).complete("sys", "user")
    assert len(sdk.calls) == 1


def test_agents_see_a_rate_limit_failure_as_a_degraded_path(mandate_free=None):
    """A raised rate limit still reaches the agent, which degrades rather than crashes."""
    from aegis.agents.critic import CriticAgent

    sdk = StubGeminiSDK(errors=[rate_limited() for _ in range(10)])
    critique = CriticAgent(gemini(sdk, max_attempts=2)).review(
        SimpleNamespace(
            legs=(), proposal_id="p", underlying="SPY", structure="s",
            quantity=1, limit_price=1, max_loss_usd=1, thesis="t",
        ),
        {},
    )
    assert critique.passed is False
    assert "could not run" in critique.summary()


# --------------------------------------------------------------------------
# the real SDK constructors (no network: clients are lazy)
# --------------------------------------------------------------------------


def test_gemini_constructs_against_the_real_sdk():
    """Stubs never exercise the constructor, so a wrong signature would hide here."""
    client = GeminiClient("fake-key-not-used")
    assert callable(client._sdk.models.generate_content)
    assert client.model == "gemini-3.6-flash"


def test_anthropic_constructs_against_the_real_sdk():
    pytest.importorskip("anthropic")
    client = AnthropicClient("fake-key-not-used")
    assert callable(client._sdk.messages.create)
    assert client.model == "claude-sonnet-4-6"


# --------------------------------------------------------------------------
# display names
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model, expected",
    [
        ("gemini-3.6-flash", "Gemini 3.6 Flash"),
        ("gemini-4.0-pro", "Gemini 4.0 Pro"),
        ("claude-sonnet-4-6", "Claude Sonnet 4.6"),   # dashed version -> dotted
        ("claude-haiku-4-5", "Claude Haiku 4.5"),
        ("claude-opus-5", "Claude Opus 5"),
        ("gemini-3.6-flash-preview", "Gemini 3.6 Flash Preview"),
    ],
)
def test_display_names_come_from_the_model_id(model, expected):
    from aegis.agents.llm import _display_name

    assert _display_name(model) == expected


def test_each_backend_reports_a_name():
    assert GeminiClient(sdk=StubGeminiSDK()).name == "Gemini 3.6 Flash"
    assert AnthropicClient(sdk=StubAnthropicSDK()).name == "Claude Sonnet 4.6"
    assert NullClient().name == "no model"


def test_the_name_follows_an_overridden_model():
    """A hardcoded label would go stale the moment the model is overridden."""
    client = build_llm_client(
        {"GEMINI_API_KEY": "g", "GEMINI_MODEL": "gemini-4.0-pro"}, sdk=StubGeminiSDK()
    )
    assert client.name == "Gemini 4.0 Pro"


def test_an_unrecognised_model_id_still_yields_something_printable():
    from aegis.agents.llm import _display_name

    assert _display_name("some-future-model") == "Some Future Model"
    assert _display_name("mystery") == "Mystery"
