"""The read-only SQLite handle every route depends on.

Deliberately not :func:`nflpred.predlog.connect`: that runs the schema DDL on open,
which needs a writable database and would let a bug in a handler create tables.
This opens the same file with ``mode=ro`` so the operating system refuses a
write, and reuses :data:`nflpred.config.PREDICTIONS_DB` so there is one path to the
log in the project.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

from fastapi import Depends, HTTPException

from nflpred.config import PREDICTIONS_DB


def _uri(path: Path) -> str:
    # `file:` URI with mode=ro; the OS enforces the read-only contract, not us.
    return f"file:{path.as_posix()}?mode=ro"


def open_readonly(path: Path = PREDICTIONS_DB) -> sqlite3.Connection:
    """Open the log read-only, or 503 if it has not been created yet.

    The log is written by ``python -m nflpred.predict`` and seeded by
    ``scripts/seed_predlog.py``; until one of those has run there is nothing to
    serve, and that is an operational state rather than a bug, so it is a 503
    with an instruction rather than a 500.
    """
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"prediction log {path.name} does not exist yet - run "
                f"`uv run python -m nflpred.predict --season 2026 --week 1` or "
                f"`uv run python scripts/seed_predlog.py` to create it."
            ),
        )
    # `check_same_thread=False`: Starlette may run a sync dependency and its sync
    # endpoint on different threadpool threads, and this connection is opened
    # read-only for the life of one request, so cross-thread use is safe here.
    connection = sqlite3.connect(_uri(path), uri=True, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


def log_connection() -> Iterator[sqlite3.Connection]:
    """FastAPI dependency: a read-only connection closed when the request ends."""
    connection = open_readonly()
    try:
        yield connection
    finally:
        connection.close()


#: The dependency as a type annotation - `def route(connection: Connection)`.
Connection = Annotated[sqlite3.Connection, Depends(log_connection)]
