"""Elo ratings, built from scratch off the schedule.

One chronological pass over every game from 1999 on, emitting each team's rating
**before** the game it is about to play. That pre-game rating is the feature; the
post-game rating never leaves this module.

The ordering rule, stated once:

    the pre-game rating is recorded, and only then is the update applied.

Those two lines in :func:`_run` are the whole of what makes Elo legal as a
feature. ``tests/test_elo.py`` targets exactly that property: perturbing a game's
result must leave that game's own ``elo_pre`` untouched and must move the next
one's.

Starting in 1999 rather than 2006 is what D-2 kept the early seasons for — by
the time the feature matrix opens in 2006, every rating has seven seasons of
burn-in behind it and none of them is still sitting at its initial value.

Hyperparameters are frozen constants in :mod:`nflpred.config`, chosen by
``scripts/tune_elo.py`` on the training seasons alone (D-11). This module does
not search them.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any, Final

import polars as pl

from nflpred.config import (
    ELO_EXPANSION_INIT,
    ELO_HFA,
    ELO_INIT,
    ELO_K,
    ELO_MOV,
    ELO_SEASON_REGRESSION,
    FIRST_SEASON,
)

#: 538's damping constant. The multiplier is ``ln(margin + 1)`` scaled by
#: ``2.2 / (0.001 * diff + 2.2)``, which shrinks the update when a heavy
#: favourite wins big and enlarges it after an upset. Without it, margin of
#: victory is autocorrelated with rating and good teams run away.
_MOV_DAMPING: Final[float] = 2.2

#: Columns the pass reads. Kept explicit so a schedule change fails here rather
#: than producing quietly wrong ratings.
_NEEDED: Final[tuple[str, ...]] = (
    "game_id",
    "season",
    "week",
    "game_type",
    "gameday",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "location",
)


def expected_score(elo_team: float, elo_opponent: float, advantage: float = 0.0) -> float:
    """Probability ``team`` beats ``opponent``, with ``advantage`` rating points.

    The logistic curve every Elo system uses: a 400-point edge is 10:1.
    """
    return 1.0 / (1.0 + 10.0 ** ((elo_opponent - elo_team - advantage) / 400.0))


def _mov_multiplier(margin: int, winner_diff: float) -> float:
    """538's log-margin multiplier.

    ``margin`` is floored at one point. A tie has a margin of zero, and
    ``ln(0 + 1) == 0`` would multiply the whole update away — which would make
    ties non-events for Elo and contradict D-4's "a tie scores 0.5, which is
    Elo's native handling". Flooring gives a tie the update a one-point game
    would get, at half the surprise.

    ``winner_diff`` is the rating edge from the winning side's point of view,
    home advantage included. It is negative after an upset, which is what makes
    the multiplier exceed one there.
    """
    return math.log(max(abs(margin), 1) + 1.0) * (
        _MOV_DAMPING / (0.001 * winner_diff + _MOV_DAMPING)
    )


def _run(
    schedules: pl.DataFrame,
    *,
    k: float = ELO_K,
    hfa: float = ELO_HFA,
    mov: bool = ELO_MOV,
    init: float = ELO_INIT,
    expansion_init: float = ELO_EXPANSION_INIT,
    regression: float = ELO_SEASON_REGRESSION,
    first_season: int = FIRST_SEASON,
) -> Iterator[dict[str, Any]]:
    """Walk every game in kickoff order, yielding one record per game.

    Each record carries both the pre-game ratings (legal as features) and the
    post-game ones (used only for the end-of-season sanity dump). The split
    between what escapes and what does not is made by the callers below, so
    there is exactly one implementation of the update.

    Games with no score — future fixtures — still yield a record with pre-game
    ratings and simply do not trigger an update. That is what lets the live 2026
    path read an Elo feature off an unplayed game.
    """
    frame = schedules.select(_NEEDED)
    if frame.schema["gameday"] != pl.Date:
        frame = frame.with_columns(pl.col("gameday").str.to_date(strict=False))
    frame = frame.sort("gameday", "game_id", nulls_last=True)

    ratings: dict[str, float] = {}
    season_in_progress: int | None = None

    for game in frame.iter_rows(named=True):
        season = game["season"]

        # Between seasons every rating regresses toward the mean. Playoff games
        # carry the previous season's `season` value in nflverse, so this fires
        # at the true boundary rather than in January.
        if season_in_progress is not None and season != season_in_progress:
            for team in ratings:
                ratings[team] = init + (1.0 - regression) * (ratings[team] - init)
        season_in_progress = season

        home, away = game["home_team"], game["away_team"]
        for team in (home, away):
            if team not in ratings:
                # A team first seen after the opening season is an expansion
                # franchise (HOU 2002 is the only one in range) and starts below
                # the mean rather than at it.
                ratings[team] = init if season == first_season else expansion_init

        elo_home, elo_away = ratings[home], ratings[away]
        neutral = game["location"] == "Neutral"
        advantage = 0.0 if neutral else hfa
        expected_home = expected_score(elo_home, elo_away, advantage)

        # --- the pre-game rating is recorded here, before any update ---
        record: dict[str, Any] = {
            "game_id": game["game_id"],
            "season": season,
            "week": game["week"],
            "game_type": game["game_type"],
            "gameday": game["gameday"],
            "home_team": home,
            "away_team": away,
            "is_neutral_site": int(neutral),
            "home_elo_pre": elo_home,
            "away_elo_pre": elo_away,
            "home_elo_prob": expected_home,
        }

        home_score, away_score = game["home_score"], game["away_score"]
        if home_score is not None and away_score is not None:
            margin = int(home_score) - int(away_score)
            actual_home = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5

            if mov:
                adjusted_home = elo_home + advantage
                if margin > 0:
                    winner_diff = adjusted_home - elo_away
                elif margin < 0:
                    winner_diff = elo_away - adjusted_home
                else:
                    # No winner to take the perspective of. Using the favourite's
                    # edge damps the update, which is the right direction: a tie
                    # is less surprising for two evenly matched teams.
                    winner_diff = abs(adjusted_home - elo_away)
                multiplier = _mov_multiplier(margin, winner_diff)
            else:
                multiplier = 1.0

            # --- and only now is it applied ---
            delta = k * multiplier * (actual_home - expected_home)
            ratings[home] = elo_home + delta
            ratings[away] = elo_away - delta

        record["home_elo_post"] = ratings[home]
        record["away_elo_post"] = ratings[away]
        yield record


def build_elo(schedules: pl.DataFrame, **kwargs: float | bool) -> pl.DataFrame:
    """One row per ``(game_id, team)`` carrying pre-game ratings only.

    Mirrors the shape of :func:`nflpred.features.rolling.build_rolling` so
    :mod:`nflpred.features.build` can join the two the same way. ``elo_diff`` is
    from this team's point of view and excludes home-field advantage, which the
    model is free to learn separately; ``elo_prob`` includes it, and is the
    Elo-only estimator the spec asks for as a sixth base model.
    """
    games = pl.DataFrame(
        list(_run(schedules, **kwargs)),
        schema_overrides={"gameday": pl.Date, "week": pl.Int64},
    ).drop("home_elo_post", "away_elo_post")

    shared = ("game_id", "season", "week", "game_type", "gameday", "is_neutral_site")

    home = games.select(
        *shared,
        team=pl.col("home_team"),
        opponent=pl.col("away_team"),
        is_home=pl.lit(1, dtype=pl.Int8),
        elo_pre=pl.col("home_elo_pre"),
        elo_opp_pre=pl.col("away_elo_pre"),
        elo_prob=pl.col("home_elo_prob"),
    )
    away = games.select(
        *shared,
        team=pl.col("away_team"),
        opponent=pl.col("home_team"),
        is_home=pl.lit(0, dtype=pl.Int8),
        elo_pre=pl.col("away_elo_pre"),
        elo_opp_pre=pl.col("home_elo_pre"),
        elo_prob=1.0 - pl.col("home_elo_prob"),
    )

    return (
        pl.concat([home, away])
        .with_columns(elo_diff=pl.col("elo_pre") - pl.col("elo_opp_pre"))
        .sort("gameday", "game_id", "team")
    )


def season_end_ratings(schedules: pl.DataFrame, **kwargs: float | bool) -> pl.DataFrame:
    """Each team's rating after its last game of each season.

    Diagnostic only — post-game ratings are not features and this frame is never
    joined to anything. It exists so the face-validity check (2007 NE at the top,
    2008 DET at the bottom) can be run against a rating system that is otherwise
    only visible through its predictions.
    """
    rows: list[dict[str, Any]] = []
    for record in _run(schedules, **kwargs):
        for side in ("home", "away"):
            rows.append(
                {
                    "season": record["season"],
                    "team": record[f"{side}_team"],
                    "elo": record[f"{side}_elo_post"],
                }
            )

    # Chronological order is preserved by `_run`, so the last row per
    # (season, team) is that team's rating after its final game.
    return pl.DataFrame(rows).group_by("season", "team").agg(pl.col("elo").last())


def main() -> None:
    """Rating sanity dump: top and bottom five for a few known seasons."""
    from nflpred.ingest import read_schedules

    schedules = read_schedules()
    elo = build_elo(schedules)
    finals = season_end_ratings(schedules)

    print(f"elo: {elo.height:,} rows x {elo.width} cols")
    print(f"  seasons {elo['season'].min()}-{elo['season'].max()}")
    print(f"  elo_pre range: {elo['elo_pre'].min():.0f} .. {elo['elo_pre'].max():.0f}")
    print(f"  null elo_pre: {elo['elo_pre'].null_count()}")

    for season in (2007, 2008, 2015, 2023):
        year = finals.filter(pl.col("season") == season).sort("elo", descending=True)

        def rated(frame: pl.DataFrame) -> str:
            return ", ".join(f"{r['team']} {r['elo']:.0f}" for r in frame.iter_rows(named=True))

        top, bottom = rated(year.head(5)), rated(year.tail(5))
        print(f"\n{season} end-of-season")
        print(f"  top:    {top}")
        print(f"  bottom: {bottom}")


if __name__ == "__main__":
    main()
