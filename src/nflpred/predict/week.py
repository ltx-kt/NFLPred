"""One target week and the history it is entitled to see: the synthesis (D-27).

See the package docstring for why an unplayed game becomes one synthesized row
per team, and why the team-game table is rebuilt here rather than read (D-33).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import polars as pl

from nflpred.config import (
    MATRIX_START_SEASON,
    POSTSEASON_TYPES,
    WP_BAND,
)
from nflpred.features.build import build_features
from nflpred.features.pbp_agg import build_team_game
from nflpred.ingest import read_pbp, read_schedules
from nflpred.modeling.ensemble import ENSEMBLE_FEATURES

#: Columns :func:`_synthesized_rows` must fill itself. Everything else in the
#: team-game table is a statistic and is null for a game nobody has played.
_SPINE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "game_type",
    "gameday",
    "team",
    "opponent",
    "is_home",
    "is_neutral_site",
    "is_postseason",
)


@dataclass(frozen=True)
class WeekView:
    """One target week, and the history it is entitled to see."""

    season: int
    week: int
    league_week: int
    kickoff: date
    target: pl.DataFrame
    completed: pl.DataFrame
    matrix: pl.DataFrame
    synthesized: tuple[str, ...]

    @property
    def live(self) -> bool:
        """True when at least one game of the week has not been played."""
        return bool(self.synthesized)


# ------------------------------------------------------------- synthesis


def _synthesized_rows(week_schedule: pl.DataFrame, template: pl.DataFrame) -> pl.DataFrame:
    """Two team-game rows per unplayed fixture, statistics all null.

    Mirrors :func:`nflpred.features.pbp_agg._spine`'s output shape rather than
    calling it, because that function's job is to *exclude* these games. The
    template supplies dtypes so the concatenation below is a plain vertical
    one - a diagonal concat would paper over a column this forgot to fill.
    """
    shared = (
        pl.col("game_id"),
        pl.col("season"),
        pl.col("week"),
        pl.col("game_type"),
        pl.col("gameday"),
    )
    neutral = (pl.col("location") == "Neutral").cast(pl.Int8)
    postseason = pl.col("game_type").is_in(POSTSEASON_TYPES).cast(pl.Int8)

    sides = [
        week_schedule.select(
            *shared,
            team=pl.col(f"{side}_team"),
            opponent=pl.col(f"{other}_team"),
            is_home=pl.lit(1 if side == "home" else 0, dtype=pl.Int8),
            is_neutral_site=neutral,
            is_postseason=postseason,
        )
        for side, other in (("home", "away"), ("away", "home"))
    ]

    rows = pl.concat(sides, how="vertical")
    unknown = [c for c in template.columns if c not in _SPINE_COLUMNS]
    return rows.with_columns(
        [pl.lit(None, dtype=template.schema[c]).alias(c) for c in unknown]
    ).select([pl.col(c).cast(template.schema[c]) for c in template.columns])


def _blocking_weeks(
    schedules: pl.DataFrame, season: int, week: int, kickoff: date, teams: Sequence[str]
) -> list[int]:
    """Earlier weeks of ``season`` that ``teams`` have not played yet.

    The documented limit, as a computation. A non-empty answer means the
    requested week is not any of these teams' *next* game, so the rolling
    windows behind it would have to be simulated rather than observed.
    """
    involved = pl.col("home_team").is_in(list(teams)) | pl.col("away_team").is_in(list(teams))
    pending = schedules.filter(
        (pl.col("season") == season)
        & (pl.col("week") < week)
        & involved
        & (pl.col("gameday").str.to_date(strict=False) < kickoff)
        & (pl.col("home_score").is_null() | pl.col("away_score").is_null())
    )
    return sorted(set(pending["week"].to_list()))


def team_game_table(pbp: pl.DataFrame, schedules: pl.DataFrame) -> pl.DataFrame:
    """The garbage-time team-game table the live path predicts on, built now.

    Callers replaying many weeks build it once and pass it to every
    :func:`week_frame` call; a single run lets ``week_frame`` build it.
    """
    return build_team_game(pbp, schedules, WP_BAND)


def _missing_history(
    team_game: pl.DataFrame,
    schedules: pl.DataFrame,
    pbp: pl.DataFrame,
    season: int,
    kickoff: date,
) -> tuple[list[str], list[str]]:
    """Completed games before ``kickoff`` the week's history does not contain.

    Returns ``(no_row, no_pbp)``: games with a score in the schedule but no row
    in ``team_game``, and games with a score but no play-by-play at all. The
    second is checked against the play-by-play rather than the table's
    statistics because a row from ``_spine`` exists for every scored game, and
    garbage-time aggregates are legitimately null for a side that ran no plays
    inside the win-probability band.

    Scoped to the matrix seasons: three 1999-2000 games have never had
    play-by-play in nflverse, and no feature row reads that far back through a
    rolling window.
    """
    history = schedules.filter(
        pl.col("season").is_between(MATRIX_START_SEASON, season)
        & pl.col("home_score").is_not_null()
        & pl.col("away_score").is_not_null()
        & (pl.col("gameday").str.to_date(strict=False) < kickoff)
    ).select("game_id")

    def absent(frame: pl.DataFrame) -> list[str]:
        present = frame.select("game_id").unique()
        return sorted(history.join(present, on="game_id", how="anti")["game_id"].to_list())

    return absent(team_game), absent(pbp)


def _preview(game_ids: Sequence[str], limit: int = 5) -> str:
    shown = ", ".join(game_ids[:limit])
    more = len(game_ids) - limit
    return f"{shown} and {more} more" if more > 0 else shown


def week_frame(
    season: int,
    week: int,
    team_game: pl.DataFrame | None = None,
    schedules: pl.DataFrame | None = None,
    pbp: pl.DataFrame | None = None,
) -> WeekView:
    """The feature rows for one week, whether or not it has been played.

    A week with results is a slice of the built matrix. A week without them is
    synthesized first - but through the *same* ``build_features`` call, so there
    is one construction of a feature row in the project and not two. A week that
    is half played (a Thursday game done, the Sunday slate not) synthesizes only
    the games that are missing, which falls out of the same code.

    ``team_game`` defaults to :func:`team_game_table` over ``pbp`` and
    ``schedules``, never to the parquet snapshot (D-33). A table passed in is
    checked against the schedule like any other.
    """
    schedules = read_schedules() if schedules is None else schedules
    pbp = read_pbp() if pbp is None else pbp
    team_game = team_game_table(pbp, schedules) if team_game is None else team_game

    week_schedule = schedules.filter((pl.col("season") == season) & (pl.col("week") == week))
    if not week_schedule.height:
        msg = f"no games scheduled for {season} week {week}."
        raise ValueError(msg)

    kickoff = week_schedule["gameday"].str.to_date(strict=False).min()

    no_row, no_pbp = _missing_history(team_game, schedules, pbp, season, kickoff)
    if no_row:
        msg = (
            f"{season} week {week}: the team-game table is missing {len(no_row)} completed "
            f"games that kicked off before it ({_preview(no_row)}). The table is older "
            f"than the schedule; predicting would silently drop those games from every "
            f"rolling window. Build it from the current cache with "
            f"`nflpred.predict.team_game_table` instead of passing a stale one."
        )
        raise ValueError(msg)
    if no_pbp:
        msg = (
            f"{season} week {week}: {len(no_pbp)} completed games before it have a score "
            f"but no play-by-play ({_preview(no_pbp)}). nflverse usually publishes "
            f"play-by-play within a day of the final whistle; re-run "
            f"`uv run python -m nflpred.ingest --seasons {season} --force` once it is up."
        )
        raise ValueError(msg)

    unplayed = week_schedule.filter(pl.col("home_score").is_null() | pl.col("away_score").is_null())

    if unplayed.height:
        teams = sorted(set(unplayed["home_team"].to_list() + unplayed["away_team"].to_list()))
        blocking = _blocking_weeks(schedules, season, week, kickoff, teams)
        if blocking:
            msg = (
                f"{season} week {week} is not the next unplayed week for these teams: "
                f"weeks {blocking} are still unplayed and sit before it. Predicting it "
                f"would require simulating those weeks to know what form each team "
                f"carries in, which this system does not do. Predict week "
                f"{blocking[0]} instead."
            )
            raise ValueError(msg)
        team_game = pl.concat([team_game, _synthesized_rows(unplayed, team_game)], how="vertical")

    matrix = build_features(team_game, schedules, pbp)
    target = matrix.filter((pl.col("season") == season) & (pl.col("week") == week)).sort(
        "gameday", "game_id"
    )
    if not target.height:
        msg = (
            f"{season} week {week} produced no feature rows. The season is before "
            f"the matrix start, or the schedule and the team-game table disagree."
        )
        raise ValueError(msg)

    missing = [c for c in ENSEMBLE_FEATURES if target[c].null_count()]
    if missing:
        msg = (
            f"{season} week {week} has null features {missing}. Every rolling window "
            f"is shift(1)ed and the synthesized row is last in its team's series, so "
            f"a null here means that invariant broke - investigate before predicting."
        )
        raise ValueError(msg)

    completed = matrix.filter(
        (pl.col("gameday") < kickoff) & pl.col("home_win").is_not_null()
    ).sort("gameday", "game_id")

    return WeekView(
        season=season,
        week=week,
        league_week=int(target["league_week"][0]),
        kickoff=kickoff,
        target=target,
        completed=completed,
        matrix=matrix,
        synthesized=tuple(unplayed["game_id"].to_list()),
    )
