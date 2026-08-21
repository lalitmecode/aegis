"""End-to-end run against real Alpaca paper trading.

Walks the mandate's universe one symbol at a time: fetches live portfolio
state, proposes a spread from the live option chain, has the critic argue with
it, shows the risk guard's verdict clause by clause, and then stops and asks a
human.

There is deliberately no --yes flag and no auto-approve path. The human gate is
the product, not a formality: skip is the default on every prompt, and an order
reaches Alpaca only after two explicit confirmations.

    python run_live.py
    python run_live.py --symbols SPY QQQ --dte 30
"""

from __future__ import annotations

import argparse
import getpass
import logging
import os
import sys
from datetime import date, datetime, timezone
from decimal import Decimal
from functools import partial
from pathlib import Path
from typing import Any

import yaml
from alpaca.data.requests import OptionSnapshotRequest
from alpaca.trading.enums import AssetClass
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from aegis.agents.critic import CriticAgent
from aegis.agents.llm import build_llm_client
from aegis.agents.research import ResearchAgent
from aegis.core.approval import APPROVAL_TTL_SECONDS, ApprovalToken
from aegis.core.audit import HashChainAudit
from aegis.core.gateway import ExecutionGateway, GovernanceError
from aegis.core.option_chain import Clients, fetch_chain
from aegis.core.proposal import PortfolioState, parse_occ_symbol
from aegis.core.risk import RiskGuard, derive_max_loss, net_delta
from aegis.core.session import EASTERN, AlpacaMarketSession
from aegis.core.verifier import ObservationVerifier

MANDATE_PATH = Path(__file__).parent / "config" / "mandate.yaml"
AUDIT_DIR = Path(__file__).parent / "logs"
REQUIRED_OPTIONS_LEVEL = 3

#: Third-party loggers that chatter through a normal run -- request traces,
#: connection pool churn, SDK notices. Silenced to ERROR so the console shows
#: Aegis's own decisions and nothing else. `google_genai` uses an underscore,
#: not `google.genai`, and the name covers its children by hierarchy.
NOISY_LOGGERS = ("google_genai", "httpx", "urllib3")

console = Console()


# --------------------------------------------------------------------------
# live portfolio state
# --------------------------------------------------------------------------


class LivePortfolio:
    """Fetches current portfolio state from Alpaca.

    This is the gateway's ``PortfolioSource``: it is called again inside
    ``submit()`` so the risk guard always evaluates a fresh snapshot, not the
    one the operator was looking at when they approved.
    """

    def __init__(self, clients: Clients) -> None:
        self._clients = clients

    def fetch(self) -> PortfolioState:
        account = self._clients.trading.get_account()
        positions = self._clients.trading.get_all_positions() or []
        equity = Decimal(str(account.equity))

        by_symbol: dict[str, int] = {}
        for position in positions:
            underlying = _underlying_of(position)
            by_symbol[underlying] = 1  # one logical position per symbol; see README

        capital_at_risk = Decimal(str(account.maintenance_margin or 0))
        return PortfolioState(
            equity=equity,
            buying_power=Decimal(str(account.buying_power)),
            open_positions=len(by_symbol),
            positions_by_symbol=by_symbol,
            portfolio_delta=float(self._portfolio_delta(positions)),
            capital_at_risk_pct=float(capital_at_risk / equity * 100) if equity > 0 else 0.0,
            fetched_at=datetime.now(timezone.utc),
        )

    def _portfolio_delta(self, positions) -> Decimal:
        """Share-equivalent delta across open option positions.

        Refuses to guess: an option position whose delta cannot be observed
        aborts the run rather than being silently counted as zero, which would
        understate portfolio delta and could let the guard pass a trade it
        should refuse.
        """
        option_symbols = [
            p.symbol for p in positions if p.asset_class == AssetClass.US_OPTION
        ]
        if not option_symbols:
            return Decimal(0)

        snapshots = self._clients.option.get_option_snapshot(
            OptionSnapshotRequest(symbol_or_symbols=option_symbols)
        )
        total = Decimal(0)
        for position in positions:
            if position.asset_class != AssetClass.US_OPTION:
                continue
            greeks = getattr(snapshots.get(position.symbol), "greeks", None)
            delta = getattr(greeks, "delta", None)
            if delta is None:
                raise RuntimeError(
                    f"no observed delta for open position {position.symbol}; "
                    "refusing to run with an understated portfolio delta"
                )
            total += Decimal(str(delta)) * Decimal(str(position.qty)) * 100
        return total


def _underlying_of(position) -> str:
    if position.asset_class == AssetClass.US_OPTION:
        try:
            return parse_occ_symbol(position.symbol).root
        except ValueError:
            return position.symbol
    return position.symbol


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def render_account(account, session_state) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    table.add_row("account", f"{account.account_number}  ({account.status})")
    table.add_row("equity", f"${Decimal(str(account.equity)):,.2f}")
    table.add_row("buying power", f"${Decimal(str(account.buying_power)):,.2f}")
    table.add_row("options level", str(account.options_trading_level))
    table.add_row(
        "market",
        "[green]open[/green]" if session_state.is_open else "[red]closed[/red]",
    )
    if session_state.opened_at:
        table.add_row("session opened", session_state.opened_at.isoformat())
    console.print(Panel(table, title="Alpaca paper account", border_style="cyan"))


def render_portfolio(state: PortfolioState) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(justify="right")
    table.add_row("open positions", str(state.open_positions))
    table.add_row("held symbols", ", ".join(state.positions_by_symbol) or "none")
    table.add_row("portfolio delta", f"{state.portfolio_delta:,.1f}")
    table.add_row("capital at risk", f"{state.capital_at_risk_pct:.2f}%")
    console.print(Panel(table, title="Portfolio state", border_style="cyan"))


def render_proposal(proposal, llm) -> None:
    legs = Table(box=None, pad_edge=False)
    legs.add_column("side")
    legs.add_column("contract")
    legs.add_column("strike", justify="right")
    legs.add_column("delta", justify="right")
    legs.add_column("open int.", justify="right")
    for leg in proposal.legs:
        parsed = parse_occ_symbol(leg.symbol)
        legs.add_row(
            leg.side.value.upper(),
            leg.symbol,
            f"{parsed.strike:,.0f}",
            f"{leg.delta}",
            f"{leg.open_interest:,}" if leg.open_interest is not None else "—",
        )

    facts = Table.grid(padding=(0, 2))
    facts.add_column(style="dim")
    facts.add_column(justify="right")
    facts.add_row("structure", proposal.structure)
    facts.add_row("quantity", str(proposal.quantity))
    facts.add_row("net credit", f"${proposal.limit_price} per spread")
    facts.add_row("max loss", f"${proposal.max_loss_usd}")
    facts.add_row("proposal hash", proposal.content_hash()[:16] + "...")

    console.print(Panel(legs, title=f"Proposal {proposal.proposal_id}", border_style="white"))
    console.print(facts)
    if proposal.thesis:
        author = llm.name if llm is not None else "a model"
        console.print(Panel(Text(proposal.thesis), title=f"Thesis (written by {author})",
                            border_style="blue"))
    else:
        reason = "no LLM provider configured" if llm is None else "the model call failed"
        console.print(f"[yellow]No thesis: {reason}. The trade is unaffected.[/yellow]")


def render_critique(critique) -> None:
    if critique.passed:
        body = Text("No concerns raised.\n"
                    "This is not an approval — the critic has no mechanism to approve.",
                    style="dim")
        console.print(Panel(body, title="Critic", border_style="blue"))
        return

    table = Table(box=None, pad_edge=False, show_header=False)
    for concern in critique.concerns:
        table.add_row("•", Text(concern))
    if critique.clause_refs:
        table.add_row("", Text("clauses: " + ", ".join(critique.clause_refs), style="dim"))
    console.print(Panel(table, title="Critic concerns (advisory)", border_style="yellow"))


#: (mandate path, substring that identifies the clause in a guard reason)
CLAUSE_ROWS = (
    ("risk_limits.max_loss_per_trade_usd", "max_loss_per_trade_usd"),
    ("risk_limits.max_loss_per_trade_pct_of_equity", "max_loss_per_trade_pct_of_equity"),
    ("risk_limits.max_open_positions", "max_open_positions"),
    ("risk_limits.max_positions_per_symbol", "max_positions_per_symbol"),
    ("risk_limits.max_portfolio_delta_abs", "max_portfolio_delta_abs"),
    ("risk_limits.max_total_capital_at_risk_pct", "max_total_capital_at_risk_pct"),
    ("risk_limits.min_buying_power_buffer_usd", "min_buying_power_buffer_usd"),
    ("strategy.delta_limits.short_leg_abs_delta_max", "short_leg_abs_delta_max"),
    ("strategy.min_credit_to_max_loss_ratio", "min_credit_to_max_loss_ratio"),
    ("strategy.expiry_window_days", "expiry_window_days"),
    ("universe.min_open_interest", "open interest"),
    ("timing.market_hours_only", "market is closed"),
    ("timing.no_new_positions_after", "no_new_positions_after"),
    ("timing.no_new_positions_within_minutes_of_open", "minutes of the open"),
)


def render_decision(decision, proposal, state, mandate, session_state) -> None:
    """Every clause the guard enforces, with the mandate's value and ours.

    Pass/fail comes from the guard's own reasons rather than being recomputed
    here, so this table can never disagree with the verdict it is describing.
    """
    reasons = " | ".join(decision.reasons)
    limits = mandate.get("risk_limits") or {}
    strategy = mandate.get("strategy") or {}
    actuals = _actuals(proposal, state, session_state)

    table = Table(box=None, pad_edge=False)
    table.add_column("")
    table.add_column("clause", style="dim", overflow="fold")
    table.add_column("mandate", justify="right")
    table.add_column("this trade", justify="right")

    flat = {**{f"risk_limits.{k}": v for k, v in limits.items()},
            "strategy.delta_limits.short_leg_abs_delta_max":
                (strategy.get("delta_limits") or {}).get("short_leg_abs_delta_max"),
            "strategy.min_credit_to_max_loss_ratio":
                strategy.get("min_credit_to_max_loss_ratio"),
            "strategy.expiry_window_days":
                _window(strategy.get("expiry_window_days") or {}),
            "universe.min_open_interest":
                (mandate.get("universe") or {}).get("min_open_interest"),
            "timing.market_hours_only": (mandate.get("timing") or {}).get("market_hours_only"),
            "timing.no_new_positions_after":
                (mandate.get("timing") or {}).get("no_new_positions_after"),
            "timing.no_new_positions_within_minutes_of_open":
                (mandate.get("timing") or {}).get("no_new_positions_within_minutes_of_open")}

    for path, needle in CLAUSE_ROWS:
        failed = needle in reasons
        table.add_row(
            "[red]✗[/red]" if failed else "[green]✓[/green]",
            path,
            str(flat.get(path, "—")),
            actuals.get(path, "—"),
        )
    console.print(Panel(table, title="Risk guard — every clause evaluated",
                        border_style="green" if decision.approved else "red"))

    if decision.approved:
        console.print("[green]Risk guard: APPROVED[/green]")
    else:
        console.print("[red]Risk guard: REFUSED[/red]")
        for reason in decision.reasons:
            console.print(f"  [red]•[/red] {reason}")


def _eastern_now() -> datetime:
    return datetime.now(timezone.utc).astimezone(EASTERN)


def _since_open(session_state) -> str:
    """Minutes elapsed since this session opened, or why that is not a number."""
    if session_state.opened_at is None:
        return "no session today"
    elapsed = (datetime.now(timezone.utc) - session_state.opened_at).total_seconds() / 60
    return f"{elapsed:,.0f} min" if elapsed >= 0 else "not yet open"


def _window(window: dict) -> str:
    return f"{window.get('min')}–{window.get('max')} days"


def _actuals(proposal, state: PortfolioState, session_state) -> dict[str, str]:
    """This trade's figures, computed with the guard's own helpers."""
    loss = derive_max_loss(proposal) or Decimal(proposal.max_loss_usd)
    equity = Decimal(str(state.equity))
    delta = net_delta(proposal)
    credit = Decimal(str(proposal.limit_price)) * 100 * proposal.quantity
    short_deltas = [
        abs(leg.delta) for leg in proposal.legs if leg.is_short and leg.delta is not None
    ]
    expiry = parse_occ_symbol(proposal.legs[0].symbol).expiration
    dte = (expiry - date.today()).days

    return {
        "risk_limits.max_loss_per_trade_usd": f"${loss}",
        "risk_limits.max_loss_per_trade_pct_of_equity":
            f"{loss / equity * 100:.2f}%" if equity else "—",
        "risk_limits.max_open_positions": str(state.open_positions + 1),
        "risk_limits.max_positions_per_symbol":
            str(state.positions_by_symbol.get(proposal.underlying, 0) + 1),
        "risk_limits.max_portfolio_delta_abs":
            f"{Decimal(str(state.portfolio_delta)) + delta:,.1f}" if delta is not None else "—",
        "risk_limits.max_total_capital_at_risk_pct":
            f"{Decimal(str(state.capital_at_risk_pct)) + loss / equity * 100:.2f}%"
            if equity else "—",
        "risk_limits.min_buying_power_buffer_usd":
            f"${Decimal(str(state.buying_power)) - loss:,.0f}",
        "strategy.delta_limits.short_leg_abs_delta_max":
            f"{max(short_deltas)}" if short_deltas else "—",
        "strategy.min_credit_to_max_loss_ratio":
            f"{credit / loss:.3f}" if loss else "—",
        "strategy.expiry_window_days": f"{dte} days",
        "universe.min_open_interest": "checked live by the verifier",
        "timing.market_hours_only": "open" if session_state.is_open else "closed",
        "timing.no_new_positions_after": f"{_eastern_now():%H:%M} ET",
        "timing.no_new_positions_within_minutes_of_open": _since_open(session_state),
    }


def render_audit(audit: HashChainAudit) -> None:
    """Print the chain and its verification. Every decision, not just orders."""
    if not len(audit):
        console.print("[dim]Audit chain is empty: nothing was decided this run.[/dim]")
        return
    chain = Table(box=None, pad_edge=False)
    chain.add_column("#", style="dim", justify="right")
    chain.add_column("event")
    chain.add_column("digest", style="dim")
    for entry in audit.entries:
        chain.add_row(str(entry.seq), entry.event, entry.digest[:16] + "...")
    console.print(Panel(chain, title="Audit chain", border_style="cyan"))

    intact, problem = audit.verify()
    if intact:
        console.print(f"[green]Audit chain verified: {len(audit)} entries, "
                      f"head {audit.head[:16]}...[/green]")
    else:
        console.print(f"[red]AUDIT CHAIN BROKEN: {problem}[/red]")


def render_result(result, audit: HashChainAudit) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    table.add_row("broker order id", f"[bold green]{result.broker_order_id}[/bold green]")
    table.add_row("client order id", result.client_order_id)
    table.add_row("status", result.status)
    table.add_row("operator", result.operator)
    table.add_row("submitted at", result.submitted_at.isoformat())
    console.print(Panel(table, title="Order accepted by Alpaca", border_style="green"))

    render_audit(audit)


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------


def build_llm(no_llm: bool) -> Any | None:
    if no_llm:
        console.print("[yellow]--no-llm: running without a model. The trade is unaffected; "
                      "there will be no thesis and the critic will fail closed.[/yellow]")
        return None

    client = build_llm_client()
    if client is None:
        console.print("[yellow]No LLM provider configured (set ANTHROPIC_API_KEY or "
                      "GEMINI_API_KEY). Proceeding without one: no thesis, and the critic "
                      "fails closed.[/yellow]")
        return None
    console.print(f"[dim]LLM provider: {client.name} ({client.model})[/dim]")
    return client


def _configure_logging() -> None:
    """Aegis at INFO, the noisy third parties off, everything else at WARNING.

    Root stays at WARNING rather than ERROR so a genuine warning from a library
    we have not named still reaches the operator -- silence is the wrong
    default for a system whose whole job is surfacing refusals.
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, show_time=False)],
        force=True,  # basicConfig is a no-op if root already has handlers
    )
    logging.getLogger("aegis").setLevel(logging.INFO)
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)


def _load_env() -> None:
    """Load .env before anything reads it.

    preflight() gates on ALPACA_PAPER_TRADE, and Clients.from_env() is what
    normally loads the file -- but that runs *after* preflight. Without this,
    ALPACA_PAPER_TRADE=false in .env sails straight past the live-trading
    guard, because os.environ has not seen it yet.
    """
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        from dotenv import load_dotenv

        load_dotenv(env_file)


def preflight(mandate) -> None:
    if os.environ.get("ALPACA_PAPER_TRADE", "true").strip().lower() == "false":
        console.print("[bold red]ALPACA_PAPER_TRADE is false. This script only runs against "
                      "paper trading. Aborting.[/bold red]")
        sys.exit(1)
    if (mandate.get("mandate") or {}).get("account_type") != "paper":
        console.print("[bold red]mandate.account_type is not 'paper'. Aborting.[/bold red]")
        sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbols", nargs="*", help="subset of the mandate universe")
    parser.add_argument("--dte", type=int, default=30, help="target days to expiry")
    parser.add_argument("--no-llm", action="store_true", help="skip Claude entirely")
    args = parser.parse_args()

    _configure_logging()

    _load_env()
    mandate = yaml.safe_load(MANDATE_PATH.read_text())
    preflight(mandate)

    clients = Clients.from_env()
    account = clients.trading.get_account()
    session = AlpacaMarketSession(clients.trading)
    session_state = session.state(datetime.now(timezone.utc))
    render_account(account, session_state)

    if int(account.options_trading_level) < REQUIRED_OPTIONS_LEVEL:
        console.print(f"[bold red]Options level {account.options_trading_level} cannot trade "
                      f"spreads; level {REQUIRED_OPTIONS_LEVEL} required. Aborting.[/bold red]")
        return 1
    if not session_state.is_open:
        console.print("[yellow]Market is closed. The mandate sets timing.market_hours_only, "
                      "so the risk guard will refuse every proposal — that refusal is the "
                      "demo.[/yellow]")

    portfolio = LivePortfolio(clients)
    state = portfolio.fetch()
    render_portfolio(state)

    llm = build_llm(args.no_llm)
    chain_fetcher = partial(fetch_chain, clients=clients)
    guard = RiskGuard(mandate, session=session)
    audit_path = AUDIT_DIR / f"audit-{date.today():%Y%m%d}.jsonl"
    audit = HashChainAudit(audit_path)
    gateway = ExecutionGateway(
        mandate,
        guard,
        audit,
        clients.trading,
        verifier=ObservationVerifier.from_mandate(mandate, chain_fetcher),
    )
    research = ResearchAgent(mandate, chain_fetcher, llm, target_dte=args.dte)
    critic = CriticAgent(llm)
    operator = os.environ.get("AEGIS_OPERATOR") or getpass.getuser()

    universe = args.symbols or (mandate.get("universe") or {}).get("allowed_symbols") or []
    submitted = 0

    for symbol in universe:
        console.rule(f"[bold]{symbol}")
        try:
            proposal = research.propose(symbol, state)
        except Exception as exc:
            console.print(f"[red]{symbol}: research failed: {type(exc).__name__}: {exc}[/red]")
            continue

        if proposal is None:
            console.print(f"[dim]{symbol}: no compliant trade. "
                          f"'No trade' is a correct outcome.[/dim]")
            continue

        render_proposal(proposal, llm)
        audit.record("PROPOSED", {
            "proposal_id": proposal.proposal_id,
            "proposal_hash": proposal.content_hash(),
            "underlying": proposal.underlying,
            "quantity": proposal.quantity,
            "max_loss_usd": str(proposal.max_loss_usd),
            "has_thesis": proposal.thesis is not None,
        })

        critique = critic.review(proposal, mandate)
        render_critique(critique)
        audit.record("CRITIQUED", {
            "proposal_id": proposal.proposal_id,
            "passed": critique.passed,
            "concerns": list(critique.concerns),
            "clause_refs": list(critique.clause_refs),
        })

        decision = guard.evaluate(proposal, state)
        render_decision(decision, proposal, state, mandate, session_state)
        audit.record(
            "GUARD_APPROVED" if decision.approved else "GUARD_REFUSED",
            {"proposal_id": proposal.proposal_id, "reasons": list(decision.reasons)},
        )
        if not decision.approved:
            console.print("[dim]Not offered for approval: the guard refused it.[/dim]")
            continue

        console.print(f"\n[bold]Approval expires {APPROVAL_TTL_SECONDS}s after you grant it. "
                      f"The gateway re-runs the guard against a fresh snapshot.[/bold]")
        choice = Prompt.ask(
            f"[bold]Decision for {proposal.proposal_id}[/bold]",
            choices=["approve", "reject", "skip"],
            default="skip",
        )
        audit.record("OPERATOR_DECISION", {
            "proposal_id": proposal.proposal_id,
            "operator": operator,
            "decision": choice,
        })
        if choice != "approve":
            console.print(f"[dim]{choice}: nothing submitted.[/dim]")
            continue
        if not Confirm.ask(
            f"Submit {proposal.quantity}x {proposal.underlying} "
            f"{proposal.structure} to Alpaca?",
            default=False,
        ):
            console.print("[dim]Not confirmed: nothing submitted.[/dim]")
            continue

        token = ApprovalToken.issue(proposal, operator)
        try:
            result = gateway.submit(proposal, portfolio, token)
        except GovernanceError as exc:
            console.print(Panel(Text(str(exc)), title=f"Refused by the gateway "
                                                      f"({type(exc).__name__})",
                                border_style="red"))
            console.print("[dim]The human said yes; the gateway said no. "
                          "That is the design.[/dim]")
            continue

        render_result(result, audit)
        submitted += 1
        state = portfolio.fetch()  # positions changed; re-baseline for the next symbol

    console.rule()
    render_audit(audit)
    console.print(f"Done. {submitted} order(s) submitted; "
                  f"audit written to {audit_path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
