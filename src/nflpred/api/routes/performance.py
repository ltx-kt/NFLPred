"""Diagnostics: scoreboard, calibration, per-member Brier - all from the log."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from nflpred.api import analytics
from nflpred.api.db import Connection
from nflpred.api.schemas import Calibration, Members, Performance
from nflpred.predlog import config_suffixes, rolling_brier

router = APIRouter()

_MEMBERS_NOTE = (
    "Per-member Brier is a monitoring signal only - never fed back into the "
    "ensemble weights (spec in-season item 3). Scoped to one config suffix: "
    "pooling recipes would score a model that never existed."
)

Config = Annotated[str | None, Query(description="config suffix; omit for the live recipe")]
Season = Annotated[int | None, Query(description="restrict to one season")]


@router.get("/performance", response_model=Performance)
def performance(connection: Connection, config: Config = None, season: Season = None) -> dict:
    """Ensemble vs Elo vs de-vigged market over settled games.

    With no ``config`` the walk-forward suffixes are pooled into the one live
    recipe (matching `scripts/phase7_pipeline.py`'s headline); a different
    recipe such as the unweighted 2025 experiment is excluded. Pass a suffix to
    scope to a single fit.
    """
    return analytics.performance(connection, config=config, season=season)


@router.get("/calibration", response_model=Calibration)
def calibration(
    connection: Connection,
    config: Config = None,
    season: Season = None,
    bins: Annotated[int, Query(ge=3, le=20)] = 10,
) -> dict:
    """Reliability points for the ensemble's calibrated probability (quantile bins)."""
    return analytics.calibration(connection, config=config, season=season, bins=bins)


@router.get("/members", response_model=Members)
def members(
    connection: Connection,
    window: Annotated[int, Query(ge=1, le=52)] = 4,
    config: Config = None,
) -> dict:
    """Per-member rolling Brier, overall and over the trailing ``window`` weeks.

    Delegates to `predlog.rolling_brier`, keeping its deliberate default of the
    most recently written suffix only.
    """
    suffixes = config_suffixes(connection)
    chosen = config or (suffixes[0] if suffixes else None)
    overall, trailing = rolling_brier(connection, window=window, suffix=chosen)
    return {
        "config": chosen or "none",
        "window": window,
        "overall": overall.to_dicts() if not overall.is_empty() else [],
        "trailing": trailing.to_dicts() if not trailing.is_empty() else [],
        "note": _MEMBERS_NOTE,
    }
