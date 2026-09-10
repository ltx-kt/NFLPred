"""The Tier 1 QB composite, on a definition that survives contact with 2026.

The open question this module closes (D-8) is not "how good is the quarterback"
but "*which* quarterback". ``home_qb_id`` / ``away_qb_id`` in ``load_schedules``
identify whoever actually took the snaps, determined after the game - so they
are unavailable for a fixture that has not kicked off, and using them in
training while serving on a projection is train/serve skew on exactly the games
where the answer matters most.

The resolution is to use a **prior-starter proxy**: a team's projected starter
for game *N* is whoever started its game *N-1*. That definition is computable
before kickoff, is identical in training and at serve time, and needs no depth
chart, injury feed, or new data source. It is wrong precisely when a team
changes quarterbacks - and that case is flagged rather than hidden.

Three stages, each lagged in its own series:

1. **Actual starter** per team-game - the passer with the most dropbacks, from
   cached play-by-play, together with his line for that game.
2. **Projected starter** - the actual starter of the team's previous game,
   ``shift(1)``ed over the team's chronological series.
3. **QB form** - that player's rolling ``qb_epa`` and ``cpoe`` over his own last
   4 and 8 starts, indexed by when each value became knowable and then joined
   onto the projected starter by a strictly-earlier as-of key.

Stage 3 borrows an expression from :mod:`nflpred.features.rolling` rather than
writing its own. One place for window arithmetic to live is the invariant the
leak tests rely on; a QB-shaped copy would be a second place for a missing lag
to hide.

**Where the lag lives, and why it moved (D-31).** Stage 3 originally used
:func:`~nflpred.features.rolling.lagged` - a window ending one start *before* the
row - and joined it to the game at the quarterback's own start. That is correct
whenever the projected starter did start, and silently stale whenever he did
not: with no row of his at that gameday, the backward join fell to his previous
start and dropped one game of his history. **On an unplayed fixture he never has
a row at that gameday**, so every live prediction would have been made on a
composite one start staler than the one the model was trained on - train/serve
skew of exactly the kind D-8 exists to rule out, on exactly the games that
matter. Measured, not argued: `tests/test_predict.py` synthesizes a completed
week as if unplayed and compares it to the real matrix row.

The fix moves the lag out of the window and into the join.
:func:`~nflpred.features.rolling.through` builds the window **inclusive** of the
start it sits on, and the as-of key is that start's gameday plus one day, so a
game reads the last row strictly before it. For a game the projected starter
did start, the two formulations pick the same set of prior starts and the
numbers are unchanged; for the ~12% of team-games where he did not, the new one
uses his most recent start instead of skipping it, which is both more accurate
and - the point - identical to what the live path can compute.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Final

import polars as pl

from nflpred.config import ROLLING_WINDOWS
from nflpred.features.rolling import _as_date, through

#: How many of a team's recent games define its "usual" starter. Eight games is
#: the same half-season the rolling form windows use.
MODAL_STARTER_WINDOW: Final[int] = 8

#: What an unknown quarterback is worth. Both statistics are zero-centred by
#: construction - EPA against an average play, completion percentage *over*
#: expected - so zero is a genuine neutral rather than an imputed mean. It is
#: still flagged: see ``qb_form_unknown``.
NEUTRAL: Final[float] = 0.0


def _starter_by_game(pbp: pl.DataFrame) -> pl.DataFrame:
    """The passer with the most dropbacks in each team-game, and his line.

    Aggregating per passer first and then taking the top one means ``qb_epa``
    and ``cpoe`` describe *that quarterback's* plays rather than the team's - a
    starter knocked out in the first quarter should not inherit his backup's
    afternoon.

    Ties on dropbacks are broken by whoever threw first, which is as close to
    "the starter" as play-by-play can get without a depth chart.
    """
    dropbacks = pbp.filter(
        (pl.col("qb_dropback") == 1)
        & pl.col("passer_player_id").is_not_null()
        & pl.col("posteam").is_not_null()
    )

    per_passer = dropbacks.group_by("game_id", "posteam", "passer_player_id").agg(
        qb_dropbacks=pl.len(),
        qb_game_epa=pl.col("qb_epa").mean(),
        qb_game_cpoe=pl.col("cpoe").mean(),
        first_play=pl.col("play_id").min(),
    )

    return (
        per_passer.sort("qb_dropbacks", "first_play", descending=[True, False])
        .group_by("game_id", "posteam", maintain_order=True)
        .first()
        .rename({"posteam": "team", "passer_player_id": "actual_qb"})
        .drop("first_play")
    )


def _modal_starter(window: int = MODAL_STARTER_WINDOW) -> pl.Expr:
    """The team's most frequent starter over its previous ``window`` games.

    Built from ``window`` explicit lags rather than a rolling aggregation
    because polars has no rolling mode over strings. Every lag is at least
    ``shift(1)``, so the current game is never part of its own answer.

    Ties are broken by sorting, which is arbitrary but deterministic - a team
    that split its last eight games between two quarterbacks has no modal
    starter worth defending either way, and ``backup_qb_starting`` is the flag
    that matters there.
    """
    lags = [pl.col("actual_qb").shift(i).over("team") for i in range(1, window + 1)]
    return (
        pl.concat_list(lags)
        .list.eval(pl.element().drop_nulls().mode().sort().first())
        .list.first()
    )


def build_qb(
    pbp: pl.DataFrame,
    team_game: pl.DataFrame,
    windows: Sequence[int] = ROLLING_WINDOWS,
) -> pl.DataFrame:
    """One row per ``(game_id, team)`` carrying pre-kickoff QB form.

    Emits the projected starter, his lagged ``qb_epa`` / ``cpoe`` over each
    window, and two flags:

    ``backup_qb_starting``
        The projected starter is not the team's modal starter over its last
        eight games - a mid-season change, whether by injury or by benching.

    ``qb_form_unknown``
        The projected starter has no prior starts at all: a rookie, or the first
        start of a career. The composite is left at :data:`NEUTRAL` **and
        flagged**, never quietly mean-imputed. This flag is a direct input to
        the Phase 6 data-quality confidence driver, which is what earns it a
        column of its own.
    """
    spine = (
        _as_date(team_game)
        .select("game_id", "season", "week", "gameday", "team")
        .sort("team", "gameday", "game_id")
    )

    # --- stage 1 + 2: who started, and who is therefore projected to start ---
    starters = spine.join(_starter_by_game(pbp), on=["game_id", "team"], how="left")
    starters = starters.with_columns(
        projected_qb=pl.col("actual_qb").shift(1).over("team"),
        modal_qb=_modal_starter(),
    )

    # --- stage 3: each quarterback's own form, in his own series ---
    #
    # Each row carries his form **through** that start, not before it, and is
    # keyed one day later. So the row is a statement about what was knowable the
    # morning after he played, and the backward join below reads the last such
    # statement that predates the fixture. See the module docstring for why the
    # lag lives in the key rather than in the window (D-31).
    player_series = (
        starters.filter(pl.col("actual_qb").is_not_null())
        .sort("actual_qb", "gameday", "game_id")
        .with_columns(
            [
                through(stat, w, partition="actual_qb").alias(f"{name}_r{w}")
                for stat, name in (("qb_game_epa", "qb_epa"), ("qb_game_cpoe", "qb_cpoe"))
                for w in windows
            ],
            # Starts *including* this one, matching the inclusive window. A game
            # reading this row is reading a count of starts that all precede it.
            qb_prior_starts=(pl.int_range(pl.len(), dtype=pl.Int32) + 1).over("actual_qb"),
        )
        .select(
            # +1 day makes the backward join strict. A quarterback cannot start
            # two games on one day, so nothing else can collide with the key.
            pl.col("gameday").dt.offset_by("1d"),
            pl.col("actual_qb").alias("projected_qb"),
            "qb_prior_starts",
            *[f"{name}_r{w}" for name in ("qb_epa", "qb_cpoe") for w in windows],
        )
        .sort("gameday")
    )

    # Backward as-of join rather than an exact one on ``game_id``, and the one
    # design choice that makes this composite servable. The match is his most
    # recent start strictly before kickoff, whether or not he goes on to start
    # this game - so a played game and the same fixture viewed before kickoff
    # resolve to the same row. Never leaky: every contributing start predates
    # the game by at least a day.
    # Both frames were just sorted by `gameday`, so the precondition holds;
    # polars simply declines to verify it when `by` groups are present and warns
    # instead. Suppressed narrowly, by message, so a genuinely unsorted frame
    # elsewhere would still be reported.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Sortedness of columns cannot be checked"
        )
        form = starters.sort("gameday").join_asof(
            player_series,
            on="gameday",
            by="projected_qb",
            strategy="backward",
        )

    unknown = pl.col(f"qb_epa_r{max(windows)}").is_null()

    return form.with_columns(
        qb_form_unknown=unknown.cast(pl.Int8),
        backup_qb_starting=(
            pl.col("projected_qb").is_not_null()
            & pl.col("modal_qb").is_not_null()
            & (pl.col("projected_qb") != pl.col("modal_qb"))
        ).cast(pl.Int8),
        qb_prior_starts=pl.col("qb_prior_starts").fill_null(0),
        **{
            f"{name}_r{w}": pl.col(f"{name}_r{w}").fill_null(NEUTRAL)
            for name in ("qb_epa", "qb_cpoe")
            for w in windows
        },
    ).select(
        "game_id",
        "season",
        "week",
        "gameday",
        "team",
        "projected_qb",
        "qb_prior_starts",
        *[f"{name}_r{w}" for name in ("qb_epa", "qb_cpoe") for w in windows],
        "backup_qb_starting",
        "qb_form_unknown",
    ).sort("gameday", "game_id", "team")


def main() -> None:
    """QB composite sanity dump."""
    from nflpred.config import TEAM_GAME_PATH
    from nflpred.ingest import read_pbp

    team_game = pl.read_parquet(TEAM_GAME_PATH)
    qb = build_qb(read_pbp(), team_game)
    matrix_era = qb.filter(pl.col("season") >= 2006)

    print(f"qb: {qb.height:,} rows x {qb.width} cols")
    print(f"  matrix seasons (2006+): {matrix_era.height:,} rows")
    print(f"  distinct projected starters: {qb['projected_qb'].n_unique():,}")
    for flag in ("backup_qb_starting", "qb_form_unknown"):
        print(f"  {flag} in 2006+: {matrix_era[flag].sum():,} ({matrix_era[flag].mean():.1%})")
    print(f"  null qb_epa_r8 in 2006+: {matrix_era['qb_epa_r8'].null_count()}")


if __name__ == "__main__":
    main()
