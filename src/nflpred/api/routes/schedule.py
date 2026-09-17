"""A season's full slate - every scheduled game, predicted or not."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from nflpred.api import queries
from nflpred.api.db import Connection
from nflpred.api.schemas import Schedule

router = APIRouter()


@router.get("/schedule/{season}", response_model=Schedule)
def season_schedule(season: int, connection: Connection) -> dict:
    """Every game of `season`, with a prediction attached where one has been made."""
    result = queries.schedule(connection, season)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"no schedule synced for {season} - run `predict --settle` first.",
        )
    return result
