"""Read-only governance console.

Shows what the operator is being asked to trust: the account the mandate is
measured against, the limits themselves, and the audit chain with its
verification recomputed on read rather than taken from the file.

Deliberately read-only. Approving a trade stays in the terminal, because
:class:`~aegis.core.gateway.ExecutionGateway` being the only path to the broker
is a claim worth keeping literally true -- a second one behind an unauthenticated
HTTP endpoint would weaken it for a convenience nobody asked for.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from aegis.core.audit import read_runs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

#: Page assets live inside the package, so they resolve from the module's own
#: location rather than from the repo layout or the working directory. A
#: `ROOT / "static"` path works only when the process runs from a source
#: checkout; this one survives an installed package and any cwd.
STATIC_DIR = HERE / "static"
MANDATE_PATH = ROOT / "config" / "mandate.yaml"
LOGS_DIR = ROOT / "logs"

#: Point the console at one specific log instead of the dated ones under
#: logs/. A deployment sets this to a committed sample; locally it stays unset
#: and the date-based lookup applies.
DECISION_LOG_ENV = "AEGIS_DECISION_LOG"


def _load_env() -> None:
    """Load .env before anything reads it.

    The credential check below runs long before Clients.from_env() would load
    the file itself, so without this a local console reports "not configured"
    while .env sits beside it holding the keys.
    """
    env_file = ROOT / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
        except ImportError:  # deployment need not carry python-dotenv
            return
        load_dotenv(env_file)


def create_app(
    *,
    clients: Any | None = None,
    portfolio: Any | None = None,
    session: Any | None = None,
    mandate_path: Path = MANDATE_PATH,
    logs_dir: Path = LOGS_DIR,
    static_dir: Path = STATIC_DIR,
    decision_log: Path | str | None = None,
) -> FastAPI:
    """Build the app. Every dependency is injectable so tests never hit the network."""
    app = FastAPI(title="Aegis governance console", docs_url=None, redoc_url=None)
    _load_env()
    log_override = decision_log or os.environ.get(DECISION_LOG_ENV) or None
    if log_override:
        log_override = Path(log_override)
        # A relative AEGIS_DECISION_LOG (render.yaml sets one) must not depend
        # on where the process was started from.
        if not log_override.is_absolute():
            log_override = ROOT / log_override
    else:
        log_override = None
    state: dict[str, Any] = {"clients": clients, "portfolio": portfolio, "session": session}

    def _clients():
        """Build Alpaca clients on first use, not at import."""
        if state["clients"] is None:
            from aegis.core.option_chain import Clients

            state["clients"] = Clients.from_env()
        return state["clients"]

    def _portfolio():
        if state["portfolio"] is None:
            from aegis.core.portfolio import LivePortfolio

            state["portfolio"] = LivePortfolio(_clients())
        return state["portfolio"]

    def _session():
        if state["session"] is None:
            from aegis.core.session import AlpacaMarketSession

            state["session"] = AlpacaMarketSession(_clients().trading)
        return state["session"]

    @app.get("/api/state")
    def read_state() -> JSONResponse:
        """Account and portfolio, as the risk guard would see them."""
        if not (os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY")):
            # Not an error: the console is designed to serve an audit trail
            # with no broker credentials at all.
            return JSONResponse({"configured": False, "reason": "no Alpaca credentials"})

        try:
            account = _clients().trading.get_account()
            market = _session().state(datetime.now(timezone.utc))
            snapshot = _portfolio().fetch()
        except Exception as exc:  # configured but unreachable -- that is a fault
            raise HTTPException(
                status_code=503, detail=f"{type(exc).__name__}: {exc}"
            ) from exc

        return JSONResponse(
            {
                "configured": True,
                "account": {
                    "number": account.account_number,
                    "status": str(account.status),
                    "equity": float(account.equity),
                    "buying_power": float(account.buying_power),
                    "options_level": int(account.options_trading_level),
                },
                "market": {
                    "is_open": market.is_open,
                    "opened_at": market.opened_at.isoformat() if market.opened_at else None,
                    "closes_at": market.closes_at.isoformat() if market.closes_at else None,
                },
                "portfolio": {
                    "open_positions": snapshot.open_positions,
                    "symbols": sorted(snapshot.positions_by_symbol),
                    "portfolio_delta": snapshot.portfolio_delta,
                    "capital_at_risk_pct": snapshot.capital_at_risk_pct,
                },
                "as_of": datetime.now(timezone.utc).isoformat(),
            }
        )

    @app.get("/api/mandate")
    def read_mandate() -> JSONResponse:
        mandate = yaml.safe_load(mandate_path.read_text())
        return JSONResponse(
            {
                "name": (mandate.get("mandate") or {}).get("name"),
                "version": (mandate.get("mandate") or {}).get("version"),
                "account_type": (mandate.get("mandate") or {}).get("account_type"),
                "risk_limits": mandate.get("risk_limits") or {},
                "strategy": mandate.get("strategy") or {},
                "universe": mandate.get("universe") or {},
                "timing": mandate.get("timing") or {},
            }
        )

    @app.get("/api/audit")
    def read_audit(day: str | None = Query(default=None)) -> JSONResponse:
        """The day's chains, each verified by recomputing its digests."""
        if log_override is not None:
            stamp, runs = "sample", read_runs(log_override)
        else:
            stamp = day or f"{date.today():%Y%m%d}"
            runs = read_runs(logs_dir / f"audit-{stamp}.jsonl")
        return JSONResponse(
            {
                "day": stamp,
                "sample": log_override is not None,
                "runs": [
                    {
                        "intact": run.intact,
                        "problem": run.problem,
                        "head": run.head,
                        "entries": [
                            {
                                "seq": e.seq,
                                "event": e.event,
                                "digest": e.digest,
                                "previous": e.previous,
                                "recorded_at": e.recorded_at,
                                "payload": e.payload,
                            }
                            for e in run.entries
                        ],
                    }
                    for run in runs
                ],
            }
        )

    @app.get("/api/days")
    def read_days() -> JSONResponse:
        """Which days have a log, newest first."""
        if log_override is not None:
            return JSONResponse({"days": ["sample"], "sample": True})
        stamps = sorted(
            (p.stem.removeprefix("audit-") for p in logs_dir.glob("audit-*.jsonl")),
            reverse=True,
        )
        return JSONResponse({"days": stamps, "sample": False})

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    return app


app = create_app()
