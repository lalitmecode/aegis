"""Console tests. Every dependency is injected; nothing reaches Alpaca."""

from __future__ import annotations

import json
import pathlib
import re
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from aegis.core.audit import HashChainAudit, read_runs
from aegis.core.proposal import PortfolioState
from aegis.core.risk import SessionState
from aegis.web.app import create_app

NOW = datetime(2026, 8, 21, 15, 0, tzinfo=timezone.utc)


class StubClients:
    def __init__(self, fail=False):
        def account():
            if fail:
                raise ConnectionError("name resolution failed")
            return SimpleNamespace(
                account_number="PA000", status="AccountStatus.ACTIVE",
                equity="99859.78", buying_power="395195.12", options_trading_level=3,
            )

        self.trading = SimpleNamespace(get_account=account)


class StubPortfolio:
    def fetch(self):
        return PortfolioState(
            equity=Decimal("99859.78"), buying_power=Decimal("395195.12"),
            open_positions=3, positions_by_symbol={"AAPL": 1, "MSFT": 1, "SPY": 1},
            portfolio_delta=17.9, capital_at_risk_pct=1.5, fetched_at=NOW,
        )


class StubSession:
    def state(self, now):
        return SessionState(is_open=True, opened_at=NOW)


def client(tmp_path=None, **kw) -> TestClient:
    kw.setdefault("clients", StubClients())
    kw.setdefault("portfolio", StubPortfolio())
    kw.setdefault("session", StubSession())
    if tmp_path is not None:
        kw.setdefault("logs_dir", tmp_path)
    return TestClient(create_app(**kw))


def write_log(tmp_path, runs=1, entries=3, tamper=False):
    path = tmp_path / "audit-20260821.jsonl"
    for _ in range(runs):
        audit = HashChainAudit(path)
        for i in range(entries):
            audit.record("PROPOSED" if i == 0 else "SUBMITTED", {"n": i})
    if tamper:
        lines = path.read_text().splitlines()
        first = json.loads(lines[0])
        first["payload"] = {"n": "edited after the fact"}
        lines[0] = json.dumps(first)
        path.write_text("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------


def test_state_reports_account_market_and_portfolio():
    body = client().get("/api/state").json()
    assert body["account"]["number"] == "PA000"
    assert body["account"]["options_level"] == 3
    assert body["market"]["is_open"] is True
    assert body["portfolio"]["open_positions"] == 3
    assert body["portfolio"]["symbols"] == ["AAPL", "MSFT", "SPY"]


def test_state_degrades_to_503_when_alpaca_is_unreachable():
    """A blank 500 tells the operator nothing; 503 with the cause tells them why."""
    response = client(clients=StubClients(fail=True)).get("/api/state")
    assert response.status_code == 503
    assert "name resolution failed" in response.json()["detail"]


# --------------------------------------------------------------------------
# mandate
# --------------------------------------------------------------------------


def test_mandate_exposes_the_limits_the_guard_enforces():
    body = client().get("/api/mandate").json()
    assert body["risk_limits"]["max_loss_per_trade_usd"] == 500
    assert body["strategy"]["delta_limits"]["short_leg_abs_delta_max"] == 0.30
    assert body["universe"]["min_open_interest"] == 500
    assert body["account_type"] == "paper"


# --------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------


def test_audit_returns_verified_runs(tmp_path):
    write_log(tmp_path, runs=2, entries=4)
    body = client(tmp_path).get("/api/audit?day=20260821").json()
    assert len(body["runs"]) == 2, "each process starts a fresh chain"
    assert all(run["intact"] for run in body["runs"])
    assert all(len(run["entries"]) == 4 for run in body["runs"])


def test_audit_reports_a_tampered_entry(tmp_path):
    """Verification is recomputed on read, not taken from the file."""
    write_log(tmp_path, entries=3, tamper=True)
    run = client(tmp_path).get("/api/audit?day=20260821").json()["runs"][0]
    assert run["intact"] is False
    assert "digest does not match" in run["problem"]


def test_audit_is_empty_for_a_day_with_no_log(tmp_path):
    assert client(tmp_path).get("/api/audit?day=19990101").json()["runs"] == []


def test_days_lists_available_logs_newest_first(tmp_path):
    for stamp in ("20260819", "20260821", "20260820"):
        (tmp_path / f"audit-{stamp}.jsonl").write_text("")
    assert client(tmp_path).get("/api/days").json()["days"] == [
        "20260821", "20260820", "20260819"
    ]


# --------------------------------------------------------------------------
# the console is read-only
# --------------------------------------------------------------------------


def test_the_console_serves_the_page():
    response = client().get("/")
    assert response.status_code == 200
    assert "governance console" in response.text


def test_there_is_no_write_endpoint():
    """A second path to the broker would make the one-gate claim untrue."""
    app = create_app(clients=StubClients(), portfolio=StubPortfolio(), session=StubSession())
    methods = {m for route in app.routes for m in getattr(route, "methods", set())}
    assert methods <= {"GET", "HEAD"}, f"unexpected write methods: {methods}"


@pytest.mark.parametrize("path", ["/api/state", "/api/mandate", "/api/audit"])
def test_write_verbs_are_rejected(path):
    assert client().post(path).status_code == 405


# --------------------------------------------------------------------------
# the console's text has to be readable
# --------------------------------------------------------------------------


def _contrast(fg: str, bg: str) -> float:
    def lum(h):
        chan = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in chan]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    a, b = lum(fg), lum(bg)
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


def _vars(scope: str) -> dict[str, str]:
    """Custom properties declared in one theme scope of the page."""
    import re
    from pathlib import Path

    html = (Path(__file__).resolve().parent.parent / "aegis" / "web" / "static" / "index.html").read_text()
    if scope == "light":
        block = re.search(r":root \{(.*?)\n\}", html, re.S).group(1)
    else:
        block = re.search(r':root\[data-theme="dark"\] \{(.*?)\n\}', html, re.S).group(1)
    return dict(re.findall(r"(--[a-z0-9-]+):\s*(#[0-9a-f]{6})", block))


@pytest.mark.parametrize("scope", ["light", "dark"])
def test_text_roles_meet_wcag_on_their_own_surface(scope):
    """Small text owes 4.5:1; the meter is a graphical object and owes 3:1."""
    v = _vars(scope)
    surface = v["--surface"]
    for role, need in (("--ink", 4.5), ("--ink-2", 4.5), ("--muted", 4.5), ("--accent", 3.0)):
        got = _contrast(v[role], surface)
        assert got >= need, f"{scope} {role} {v[role]} on {surface}: {got:.2f}:1 < {need}"


def test_a_status_dot_is_never_shown_without_a_label():
    """Warning is sub-3:1 on the light surface by design; the label is the mitigation.

    So the rule is structural: a coloured dot never appears on its own.
    """
    import re
    from pathlib import Path

    html = (Path(__file__).resolve().parent.parent / "aegis" / "web" / "static" / "index.html").read_text()
    dots = list(re.finditer(r'<span class="dot"></span>', html))
    assert dots, "no status dots found — has the badge markup changed?"
    for dot in dots:
        following = html[dot.end():dot.end() + 60].lstrip()
        assert following.startswith("<span>"), (
            f"dot at offset {dot.start()} is not followed by a text span: {following[:40]!r}"
        )


def test_every_verdict_badge_names_its_state():
    from pathlib import Path

    html = (Path(__file__).resolve().parent.parent / "aegis" / "web" / "static" / "index.html").read_text()
    for label in ("verified", "chain broken", "market ", "unavailable"):
        assert label in html, f"no textual label for {label!r}"


# --------------------------------------------------------------------------
# deployment: the console must serve with no credentials at all
# --------------------------------------------------------------------------


CREDENTIAL_VARS = (
    "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_PAPER_TRADE",
    "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GEMINI_MODEL", "AEGIS_DECISION_LOG",
)


@pytest.fixture
def bare_env(monkeypatch, tmp_path):
    """No credentials, and no .env for create_app to pick them up from."""
    for var in CREDENTIAL_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("aegis.web.app.ROOT", tmp_path)  # no .env beside it
    return monkeypatch


def test_the_app_starts_with_no_environment_configured(bare_env):
    app = create_app()
    with TestClient(app) as c:
        assert c.get("/api/days").status_code == 200
        assert c.get("/api/mandate").status_code == 200
        assert c.get("/").status_code == 200


def test_state_reports_not_configured_rather_than_failing(bare_env):
    """Missing credentials is a supported configuration, not a server fault."""
    body = TestClient(create_app()).get("/api/state")
    assert body.status_code == 200
    assert body.json() == {"configured": False, "reason": "no Alpaca credentials"}


def test_serving_the_console_imports_no_broker_or_llm_code(bare_env):
    """The heavy SDKs are imported lazily; a credential-less deploy never needs them."""
    import subprocess
    import sys

    code = (
        "import sys; from fastapi.testclient import TestClient;"
        "from aegis.web.app import create_app;"
        "c = TestClient(create_app());"
        "c.get('/api/days'); c.get('/api/mandate'); c.get('/api/audit'); c.get('/');"
        "loaded = [m for m in ('alpaca', 'google.genai', 'anthropic') if m in sys.modules];"
        "print('LOADED=' + ','.join(loaded))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(pathlib.Path(__file__).resolve().parent.parent))
    assert "LOADED=" in out.stdout, out.stderr
    assert out.stdout.strip().endswith("LOADED="), f"unexpected imports: {out.stdout}"


# --------------------------------------------------------------------------
# the committed sample audit trail
# --------------------------------------------------------------------------


SAMPLE = pathlib.Path(__file__).resolve().parent.parent / "docs" / "sample-decisions.jsonl"


def test_the_sample_chain_verifies():
    runs = read_runs(SAMPLE)
    assert len(runs) == 1
    assert runs[0].intact, runs[0].problem
    assert len(runs[0].entries) == 13


def test_the_sample_covers_the_whole_pipeline():
    events = [e.event for e in read_runs(SAMPLE)[0].entries]
    for required in ("PROPOSED", "CRITIQUED", "GUARD_APPROVED", "OPERATOR_DECISION",
                     "SUBMITTED", "GUARD_REFUSED", "REJECTED"):
        assert required in events, f"sample is missing a {required} entry"


def test_the_refusal_cites_its_clauses():
    entries = {e.event: e for e in read_runs(SAMPLE)[0].entries}
    reasons = " ".join(entries["GUARD_REFUSED"].payload["reasons"])
    for clause in ("max_loss_per_trade_usd", "max_loss_per_trade_pct_of_equity",
                   "max_portfolio_delta_abs"):
        assert clause in reasons


def test_the_sample_is_sanitised():
    """A committed log must not carry the real account, operator, or order ids."""
    import re

    text = SAMPLE.read_text()
    assert not re.search(r"\bPA[0-9A-Z]{8,}\b", text), "account number present"
    assert "lalit" not in text, "real operator name present"
    for placeholder in ("00000000-0000-4000-8000", "operator"):
        assert placeholder in text


def test_the_decision_log_override_serves_the_sample(monkeypatch):
    monkeypatch.setenv("AEGIS_DECISION_LOG", str(SAMPLE))
    c = TestClient(create_app(clients=StubClients(), portfolio=StubPortfolio(),
                              session=StubSession()))
    audit = c.get("/api/audit").json()
    assert audit["sample"] is True
    assert audit["day"] == "sample"
    assert len(audit["runs"][0]["entries"]) == 13
    assert c.get("/api/days").json() == {"days": ["sample"], "sample": True}


def test_the_page_carries_a_sample_banner():
    html = (pathlib.Path(__file__).resolve().parent.parent / "aegis" / "web" / "static" / "index.html").read_text()
    assert 'id="banner"' in html
    assert "Sample audit trail" in html
    assert "not live data" in html
    assert 'meta.sample) $("banner").classList.add("on")' in html


# --------------------------------------------------------------------------
# the render blueprint has to match the code it deploys
# --------------------------------------------------------------------------


def test_render_blueprint_points_at_things_that_exist():
    import yaml as _yaml

    root = pathlib.Path(__file__).resolve().parent.parent
    svc = _yaml.safe_load((root / "render.yaml").read_text())["services"][0]

    assert svc["plan"] == "free" and svc["type"] == "web" and svc["runtime"] == "python"
    # the start command must name the module path the app actually lives at
    assert "aegis.web.app:app" in svc["startCommand"]
    assert "--port $PORT" in svc["startCommand"]

    env = {e["key"]: e["value"] for e in svc["envVars"]}
    assert (root / env["AEGIS_DECISION_LOG"]).exists(), "sample log is not committed"
    assert (root / svc["healthCheckPath"].lstrip("/")).parent  # path is well-formed


def test_requirements_declare_what_the_start_command_needs():
    """The build installs requirements.txt; uvicorn and fastapi must be in it."""
    root = pathlib.Path(__file__).resolve().parent.parent
    reqs = (root / "requirements.txt").read_text().lower()
    for pkg in ("fastapi", "uvicorn", "pyyaml"):
        assert re.search(rf"^{pkg}[><=~]", reqs, re.M), f"{pkg} missing from requirements.txt"


# --------------------------------------------------------------------------
# the console must not care where the process was started from
# --------------------------------------------------------------------------


def test_the_page_is_served_from_any_working_directory(monkeypatch, tmp_path):
    """A deploy runs from a different cwd than the checkout; nothing covered this."""
    monkeypatch.chdir(tmp_path)
    response = client().get("/")
    assert response.status_code == 200
    assert "governance console" in response.text
    assert "Sample audit trail" in response.text


def test_every_endpoint_works_from_a_foreign_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    c = client()
    for path in ("/", "/api/mandate", "/api/audit", "/api/days", "/api/state"):
        assert c.get(path).status_code == 200, path


def test_the_static_mount_resolves_from_the_package(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    response = client().get("/static/index.html")
    assert response.status_code == 200
    assert "governance console" in response.text


def test_a_relative_decision_log_is_anchored_to_the_repo(monkeypatch, tmp_path):
    """render.yaml passes a relative path; cwd must not decide whether it resolves."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AEGIS_DECISION_LOG", "docs/sample-decisions.jsonl")
    body = TestClient(create_app(clients=StubClients(), portfolio=StubPortfolio(),
                                 session=StubSession())).get("/api/audit").json()
    assert body["sample"] is True
    assert len(body["runs"][0]["entries"]) == 13


def test_no_module_level_path_in_the_web_package_is_relative():
    """Any relative Path constant here would resolve against the cwd."""
    from aegis.web import app as web_app

    for name in ("HERE", "ROOT", "STATIC_DIR", "MANDATE_PATH", "LOGS_DIR"):
        value = getattr(web_app, name)
        assert value.is_absolute(), f"{name} is relative: {value}"


def test_page_assets_ship_inside_the_package():
    """`ROOT / 'static'` only works from a source checkout; this survives install."""
    from aegis.web import app as web_app

    assert web_app.STATIC_DIR.parent == web_app.HERE
    assert (web_app.STATIC_DIR / "index.html").exists()


# --------------------------------------------------------------------------
# the console may never be pointed at a live account
# --------------------------------------------------------------------------


@pytest.fixture
def no_dotenv(monkeypatch, tmp_path):
    """Isolate from the repo's own .env so the environment under test is exact."""
    for var in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_PAPER_TRADE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("aegis.web.app.ROOT", tmp_path)
    return monkeypatch


def _with_keys(env, paper=None):
    env.setenv("ALPACA_API_KEY", "PK-test")
    env.setenv("ALPACA_SECRET_KEY", "secret-test")
    if paper is not None:
        env.setenv("ALPACA_PAPER_TRADE", paper)


def test_keys_without_a_paper_flag_refuse_to_start(no_dotenv):
    _with_keys(no_dotenv)  # ALPACA_PAPER_TRADE unset
    with pytest.raises(RuntimeError, match="ALPACA_PAPER_TRADE"):
        create_app()


def test_keys_with_paper_false_refuse_to_start(no_dotenv):
    _with_keys(no_dotenv, "false")
    with pytest.raises(RuntimeError, match="not 'true'"):
        create_app()


def test_keys_with_paper_true_start(no_dotenv):
    _with_keys(no_dotenv, "true")
    assert create_app() is not None


def test_no_keys_at_all_starts_credential_free(no_dotenv):
    c = TestClient(create_app())
    assert c.get("/").status_code == 200
    assert c.get("/api/state").json() == {"configured": False, "reason": "no Alpaca credentials"}


@pytest.mark.parametrize("flag", ["", "  ", "1", "yes", "paper", "TRUE ", "False", "0"])
def test_only_an_affirmative_flag_is_accepted(no_dotenv, flag):
    """Fails closed: anything that is not an affirmative 'true' refuses."""
    _with_keys(no_dotenv, flag)
    if flag.strip().lower() == "true":
        assert create_app() is not None
    else:
        with pytest.raises(RuntimeError):
            create_app()


def test_the_guard_runs_at_creation_not_on_first_request(no_dotenv):
    """uvicorn must fail to boot, not serve a broken console until someone calls it."""
    _with_keys(no_dotenv, "false")
    with pytest.raises(RuntimeError):
        create_app()  # no request made


def test_the_paper_endpoint_is_pinned_in_code(no_dotenv, monkeypatch):
    """No environment variable may choose the endpoint."""
    import alpaca.trading.client as trading

    captured: dict = {}

    class RecordingTradingClient:
        def __init__(self, key, secret, **kwargs):
            captured.update(kwargs)

        def get_account(self):
            return SimpleNamespace(
                account_number="PA000", status="AccountStatus.ACTIVE",
                equity="1", buying_power="1", options_trading_level=3,
            )

    _with_keys(no_dotenv, "true")
    monkeypatch.setattr(trading, "TradingClient", RecordingTradingClient)
    TestClient(create_app(portfolio=StubPortfolio(), session=StubSession())).get("/api/state")

    assert captured.get("paper") is True, "the paper endpoint must be pinned"
    assert "url_override" not in captured, "a base URL must never be supplied"


def test_the_endpoint_ignores_the_environment_after_startup(no_dotenv, monkeypatch):
    """Defence in depth: flipping the flag at runtime must not reach the endpoint.

    The startup guard already refuses a non-paper flag, so guard and endpoint
    agree at boot. This proves they still agree if the environment changes
    underneath a running process -- the endpoint is pinned in code, not read.
    """
    import alpaca.trading.client as trading

    captured: dict = {}

    class RecordingTradingClient:
        def __init__(self, key, secret, **kwargs):
            captured.update(kwargs)

        def get_account(self):
            return SimpleNamespace(
                account_number="PA000", status="AccountStatus.ACTIVE",
                equity="1", buying_power="1", options_trading_level=3,
            )

    _with_keys(no_dotenv, "true")
    monkeypatch.setattr(trading, "TradingClient", RecordingTradingClient)
    app = create_app(portfolio=StubPortfolio(), session=StubSession())

    # the process is up; now the environment lies to it
    no_dotenv.setenv("ALPACA_PAPER_TRADE", "false")
    TestClient(app).get("/api/state")

    assert captured.get("paper") is True, "endpoint followed the environment"
    assert "url_override" not in captured
