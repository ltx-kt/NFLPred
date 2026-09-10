"""Leak-safe rolling team form.

Takes the same-game rows produced by :mod:`nflpred.features.pbp_agg` and turns them
into rows that carry only what was knowable **before** kickoff. This is the one
module where leakage can enter, so the lagging itself is confined to a single
helper - :func:`lagged` - and everything else is bookkeeping around it.

The rule, stated once:

    row for game *N* carries the mean of games *N-W* .. *N-1*, never *N*.

Two ordering details matter and are easy to get wrong:

* Sorting is by ``gameday``, not ``week``. Byes and postseason rounds make week
  numbers non-contiguous, and two teams in the same week can play three days
  apart. ``gameday`` is the actual chronology.
* Windows deliberately cross season boundaries (D-6). 2006 week 1 uses the tail
  of 2005. Seasons 1999-2005 were ingested for exactly this (D-2), so every row
  the feature matrix keeps has a full window behind it - nothing dropped,
  nothing imputed.

Alongside the fixed 4- and 8-game windows there is a season-to-date window: an
expanding mean *within* the current season, which by construction has nothing
behind it in week 1. Week 1 falls back to the cross-season 8-game value and says
so in ``std_is_fallback``, rather than carrying a null or a silent imputation.

Three window expressions live here and nowhere else, so that every place a leak
could enter is in one file the leak tests audit:

``lagged``
    The lagging one. ``rolling_mean().shift(1)`` - the row's own game is
    excluded by the shift.
``_expanding``
    The season-to-date twin of the above, same shift, same rule.
``through``
    Deliberately **not** lagged: the window ends at and includes the row. It is
    legal only because its one caller - :mod:`nflpred.features.qb` - joins it to a
    game by a strictly-earlier as-of key, so the lag lives in the join instead
    of in the window. Read its docstring before using it anywhere else.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import polars as pl

from nflpred.config import ROLLING_WINDOWS, STD_WINDOW_STATS
from nflpred.features.pbp_agg import OFFENSE_STATS

#: Shorter names for the rolled columns. Purely cosmetic: ``off_epa_r8`` reads
#: better than ``off_epa_per_play_r8``, and the differentials built on top of
#: these in :mod:`nflpred.features.build` inherit the stem. Any statistic not listed
#: keeps its own name, so :data:`~nflpred.features.pbp_agg.OFFENSE_STATS` stays the
#: single source of truth for *which* statistics exist.
_STEMS: Final[dict[str, str]] = {
    "epa_per_play": "epa",
    "success_rate": "success",
    "pass_epa_per_play": "pass_epa",
    "pass_success_rate": "pass_success",
    "rush_epa_per_play": "rush_epa",
    "rush_success_rate": "rush_success",
    "explosive_rate": "explosive",
    "third_down_rate": "third_down",
    "sack_rate": "sack",
    "qb_epa_per_dropback": "qb_epa",
}

#: Columns carried through unchanged. These describe the fixture, not its
#: result - except ``won``, which is the target and is labelled as such
#: downstream rather than being fed back in as a feature.
_KEYS: Final[tuple[str, ...]] = (
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
    "won",
)


def stem(stat: str) -> str:
    """Short name used for ``stat`` in rolled column names."""
    return _STEMS.get(stat, stat)


def rolled_columns(
    windows: Sequence[int] = ROLLING_WINDOWS,
    stats: Sequence[str] = OFFENSE_STATS,
) -> tuple[str, ...]:
    """Names of every rolled column, in emission order."""
    return tuple(
        f"{side}_{stem(s)}_r{w}" for side in ("off", "def") for s in stats for w in windows
    )


def std_columns(stats: Sequence[str] = STD_WINDOW_STATS) -> tuple[str, ...]:
    """Names of every season-to-date column, in emission order.

    ``_std`` is short for season-to-date, not standard deviation.
    """
    return tuple(f"{side}_{stem(s)}_std" for side in ("off", "def") for s in stats)


def lagged(column: str, window: int, partition: str = "team") -> pl.Expr:
    """The only lagging expression in the codebase.

    ``rolling_mean`` at row *i* covers games *i-W+1* .. *i*, which includes the
    game being predicted. ``.shift(1)`` moves the whole series forward one game
    so row *i* instead carries *i-W* .. *i-1*. **That shift is what makes the
    feature legal.** ``.over(partition)`` keeps both operations inside one
    series; the caller is responsible for having sorted by
    ``(partition, gameday)`` first, since ``over`` respects frame order within a
    group.

    ``partition`` is a team here and a ``passer_player_id`` in
    :mod:`nflpred.features.qb`. Parameterising it is what keeps the QB composite
    from growing a second lagging expression - there is one idiom to audit, and
    the leak tests audit it.

    ``min_samples=1`` yields a partial mean rather than a null for a team's
    first few games. Polars skips nulls inside the window, so a statistic that
    is undefined for one game (a team with no rush attempts, say) shrinks that
    window rather than voiding it. The first game of a series is null either
    way - there is nothing prior to average.
    """
    return (
        pl.col(column)
        .rolling_mean(window_size=window, min_samples=1)
        .shift(1)
        .over(partition)
    )


def through(column: str, window: int, partition: str = "team") -> pl.Expr:
    """:func:`lagged` without the shift: the window **includes** the row itself.

    Not a lagging expression, and it must never be joined to a game row by that
    game's own key - doing so would be exactly the leak :func:`lagged` exists to
    prevent. It exists for one caller, :func:`nflpred.features.qb.build_qb`, which
    needs a series indexed by *when a value became knowable* rather than by
    *which game may use it*, and then does the excluding in the join.

    The pattern, stated once because it is the only place it appears: a
    quarterback's row at his start on day *d* carries his form **through** *d*,
    and a game on day *G* reads the last such row with ``d < G``. That gives the
    same answer whether or not he starts the game on *G* - which is what makes
    the composite computable for a fixture that has not kicked off (D-8, D-31).
    An inclusive window plus a strictly-earlier join is one lag, not none;
    splitting it across the two halves is what buys train/serve equality.
    """
    return pl.col(column).rolling_mean(window_size=window, min_samples=1).over(partition)


def _expanding(column: str) -> pl.Expr:
    """The season-to-date twin of :func:`lagged`, and the only other one.

    Same shape and the same rule: build the running mean, then ``.shift(1)`` so
    row *i* carries games *0 .. i-1* of the season rather than *0 .. i*. **That
    shift is what makes the feature legal**, exactly as above.

    ``.over(["team", "season"])`` restarts the window each September, so the
    shift also makes the first game of every season null - correct, since there
    is no season-to-date form before the season starts. :func:`build_rolling`
    fills that hole from the 8-game window and flags it.

    Nulls are summed as zero and excluded from the denominator rather than
    voiding the whole window, which matches ``min_samples=1`` above.
    """
    value = pl.col(column)
    group = ["team", "season"]
    total = value.fill_null(0.0).cum_sum().over(group)
    observed = value.is_not_null().cum_sum().over(group)

    return (
        pl.when(observed > 0)
        .then(total / observed)
        .otherwise(None)
        .shift(1)
        .over(group)
    )


def _as_date(frame: pl.DataFrame, column: str = "gameday") -> pl.DataFrame:
    """Parse ``gameday`` to a real date so ordering is chronological."""
    if frame.schema[column] == pl.Date:
        return frame
    return frame.with_columns(pl.col(column).str.to_date())


def build_rolling(
    team_game: pl.DataFrame,
    windows: Sequence[int] = ROLLING_WINDOWS,
    stats: Sequence[str] = OFFENSE_STATS,
    std_stats: Sequence[str] = STD_WINDOW_STATS,
) -> pl.DataFrame:
    """One row per ``(game_id, team)`` holding pre-kickoff form only.

    Emits ``off_<stem>_r<W>`` and ``def_<stem>_r<W>`` for every statistic and
    window, plus ``games_in_window_r<W>``: how many prior games actually backed
    that average. The count feeds the Phase 6 data-quality confidence driver,
    where a team with three games of history should not be trusted like one
    with eight.

    Also emits ``off_<stem>_std`` / ``def_<stem>_std`` for ``std_stats``: the
    season-to-date mean, restricted to the handful of statistics where "so far
    this season" carries information the 8-game window does not. Rolling all 23
    would add 46 columns of mostly noise - a season-to-date third-down-attempt
    count says more about the calendar than about the team.

    ``games_in_season`` counts prior games *within* the season and
    ``std_is_fallback`` marks the rows where that count is zero, whose
    season-to-date columns carry the 8-game value instead. Consistent with D-6,
    and visible rather than silent.
    """
    ordered = _as_date(team_game).sort("team", "gameday", "game_id")

    lags = [
        lagged(f"{side}_{s}", w).alias(f"{side}_{stem(s)}_r{w}")
        for side in ("off", "def")
        for s in stats
        for w in windows
    ]
    season_to_date = [
        _expanding(f"{side}_{s}").alias(f"{side}_{stem(s)}_std")
        for side in ("off", "def")
        for s in std_stats
    ]

    # Row index within the team's series == number of games played before it.
    prior = pl.int_range(pl.len(), dtype=pl.Int32).over("team")
    counts = [
        pl.min_horizontal(prior, pl.lit(w, dtype=pl.Int32)).alias(f"games_in_window_r{w}")
        for w in windows
    ]
    in_season = pl.int_range(pl.len(), dtype=pl.Int32).over(["team", "season"])

    # The fallback window: the longest one, which every 2006+ row is guaranteed
    # to have full behind it (D-6), so week 1 is filled rather than null.
    fallback = max(windows)

    return (
        ordered.with_columns(
            [*lags, *season_to_date, *counts, in_season.alias("games_in_season")]
        )
        .with_columns(
            std_is_fallback=(pl.col("games_in_season") == 0).cast(pl.Int8),
            **{
                f"{side}_{stem(s)}_std": pl.coalesce(
                    f"{side}_{stem(s)}_std", f"{side}_{stem(s)}_r{fallback}"
                )
                for side in ("off", "def")
                for s in std_stats
            },
        )
        .select(
            *_KEYS,
            *[f"games_in_window_r{w}" for w in windows],
            "games_in_season",
            "std_is_fallback",
            *rolled_columns(windows, stats),
            *std_columns(std_stats),
        )
    )


def main() -> None:
    """Build the rolling table from the cached team-game table and describe it."""
    from nflpred.config import TEAM_GAME_PATH

    rolling = build_rolling(pl.read_parquet(TEAM_GAME_PATH))
    first = rolling.filter(pl.col("season") >= 2006)

    print(f"rolling: {rolling.height:,} rows x {rolling.width} cols")
    print(f"  matrix seasons (2006+): {first.height:,} rows")
    print(f"  null off_epa_r8 in 2006+: {first['off_epa_r8'].null_count()}")
    print(f"  min games_in_window_r8 in 2006+: {first['games_in_window_r8'].min()}")
    print(f"  null off_epa_std in 2006+: {first['off_epa_std'].null_count()}")
    print(
        f"  std_is_fallback in 2006+: {first['std_is_fallback'].sum():,} rows "
        f"({first['std_is_fallback'].mean():.1%})"
    )


if __name__ == "__main__":
    main()
