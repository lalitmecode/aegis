"""Run the console: ``python -m aegis.web``."""

from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run("aegis.web.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
