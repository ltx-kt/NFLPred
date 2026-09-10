"""The weekly board and the nav index."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from nflpred.api import queries
from nflpred.api.db import Connection
from nflpred.api.schemas import Health, Index, Week
from nflpred.predlog import summary

router = APIRouter()


@router.get("/health", response_model=Health)
def health(connection: Connection) -> dict:
    """Row counts and settled share - a liveness check that also proves the join."""
    return summary(connection)


@router.get("/index", response_model=Index)
def nav_index(connection: Connection) -> dict:
    """Seasons, the weeks present in each, the latest week, and the configs."""
    return queries.index(connection)


@router.get("/weeks/{season}/{week}", response_model=Week)
def week_board(season: int, week: int, connection: Connection) -> dict:
    """One week: every predicted game with votes, anchors, and outcome if settled."""
    result = queries.week(connection, season, week)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"no predictions logged for {season} week {week}.",
        )
    return result
