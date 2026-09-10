"""One game: the flat row, the full output-contract record, the outcome."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from nflpred.api import queries
from nflpred.api.db import Connection
from nflpred.api.schemas import GameDetail

router = APIRouter()


@router.get("/games/{game_id}", response_model=GameDetail)
def game_detail(
    game_id: str, connection: Connection, model_version: str | None = None
) -> dict:
    """A single game. Newest config by default; ``?model_version=`` pins a run.

    ``record`` is passed through exactly as `nflpred.modeling.ensemble` wrote it, with
    the Phase 6 ``explanation`` block present only for weeks run with
    ``--explain`` (no seeded backtest week has one).
    """
    result = queries.game(connection, game_id, model_version)
    if result is None:
        detail = (
            f"no prediction logged for game {game_id}"
            if model_version is None
            else f"no prediction for game {game_id} under model_version {model_version}"
        )
        raise HTTPException(status_code=404, detail=detail)
    return result
