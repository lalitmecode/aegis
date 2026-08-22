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

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from aegis.core.audit import read_runs

ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = ROOT / "static"
MANDATE_PATH = ROOT / "config" / "mandate.yaml"
LOGS_DIR = ROOT / "logs"


def create_app(
    *,
    clients: Any | None = None,
    portfolio: Any | None = None,
    session: Any | None = None,
    mandate_path: Path = MANDATE_PATH,
    logs_dir: Path = LOGS_DIR,
    static_dir: Path = STATIC_DIR,
) -> FastAPI:
    """Build the app. Every dependency is injectable so tests never hit the network."""
    app = FastAPI(title="Aegis governance console", docs_url=None, redoc_url=None)
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
        try:
            account = _clients().trading.get_account()
            market = _session().state(datetime.now(timezone.utc))
            snapshot = _portfolio().fetch()
        except Exception as exc:  # the console degrades; it does not 500 blankly
            raise HTTPException(
                status_code=503, detail=f"{type(exc).__name__}: {exc}"
            ) from exc

        return JSONResponse(
            {
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
        stamp = day or f"{date.today():%Y%m%d}"
        runs = read_runs(logs_dir / f"audit-{stamp}.jsonl")
        return JSONResponse(
            {
                "day": stamp,
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
        stamps = sorted(
            (p.stem.removeprefix("audit-") for p in logs_dir.glob("audit-*.jsonl")),
            reverse=True,
        )
        return JSONResponse({"days": stamps})

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    return app


app = create_app()
