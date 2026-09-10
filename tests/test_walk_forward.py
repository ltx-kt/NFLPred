"""The walk-forward harness, tested the way `test_no_leakage.py` tests features.

Phase 7 is the first code in the project permitted to name the test split
(D-29), so the AST guard that `test_models.py`, `test_ensemble.py` and
`test_explanations.py` put over their phase scripts is deliberately **not**
extended here — the harness has to be able to walk 2022-2025, and a guard that
forbade the string would forbid the deliverable.

What replaces it is stronger anyway, and is the same shape as
``test_models.py::test_fitting_ignores_the_test_split``: rewrite every feature
and every result at and after the target week to garbage, run the harness again,
and demand that week's predictions come back bit-identical. A static check says
the code does not *ask* for later rows; this says it does not *use* them, which
would also catch a join, a global mean, or a scaler that quietly pulled the
whole matrix in.

Alongside it, the chronology claim directly: the most recent game either fitted
window saw is strictly earlier than the target week's first kickoff. Cut by
``gameday``, never by week number — a Thursday game and the Monday game after it
are the same "week" and eleven days apart.

These run against the real matrix. A synthetic frame would not exercise byes,
postseason weeks, or the season boundaries where the calibration window rolls.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nflpred.backtest import (
    WEIGHT_COLUMN,
    fit_week,
    score_week,
    split_windows,
    walk_forward,
    week_boundaries,
    with_recency,
)
from nflpred.config import CALIB_WINDOW_SEASONS
from nflpred.features.build import CORE_FEATURES
from nflpred.modeling.ensemble import ENSEMBLE_FEATURES

pytestmark = pytest.mark.data

#: One validation season, walked in full. Chosen rather than sampled so a
#: failure is reproducible, and 2019 because it is the first season after the
#: frozen calibration split — the boundary where the rolling window first has to
#: move off D-5's fixed seasons.
SAMPLE_SEASON: int = 2019

#: A single week, for the tests that must fit twice. Week 8 is deep enough into
#: the season that the calibration window contains a partial season, which is
#: the D-26 case a week-1 target would not exercise.
POISON_WEEK: int = 8

#: Deliberately not `None`: the poisoning test should exercise the weighted path,
#: since a weight column computed off a poisoned `league_week` would be one more
#: way for later rows to reach a fit.
HALF_LIFE: float = 20.0


@pytest.fixture(scope="module")
def season_walk(matrix: pl.DataFrame):
    """One full season of walk-forward. The slow fixture; everything reuses it."""
    return walk_forward(
        matrix, (SAMPLE_SEASON, SAMPLE_SEASON), ENSEMBLE_FEATURES, half_life=HALF_LIFE
    )


# ------------------------------------------------------------- chronology


def test_every_fit_predates_the_week_it_predicts(season_walk):
    """The load-bearing claim, for every week of the sampled season."""
    for row in season_walk.weeks.iter_rows(named=True):
        assert row["last_fitted_kickoff"] < row["kickoff"], (
            f"{row['season']} week {row['week']}: fitted on a game from "
            f"{row['last_fitted_kickoff']}, which is not before kickoff "
            f"{row['kickoff']}"
        )


def test_the_windows_grow_and_never_overlap(season_walk):
    """Train and calibration share no season, and both roll forward (D-26)."""
    weeks = season_walk.weeks
    train_last = [int(r.split("-")[1]) for r in weeks["train_seasons"]]
    calib_first = [int(r.split("-")[0]) for r in weeks["calib_seasons"]]

    assert all(t < c for t, c in zip(train_last, calib_first, strict=True))
    assert weeks["n_train"].to_list() == sorted(weeks["n_train"].to_list())
    assert weeks["n_calib"].max() > weeks["n_calib"].min(), (
        "the calibration window never grew — it is not rolling"
    )


def test_the_calibration_window_is_three_seasons(matrix):
    """D-26's window, at a mid-season target where it spans a partial season."""
    boundaries = week_boundaries(matrix, (SAMPLE_SEASON, SAMPLE_SEASON))
    bound = boundaries.filter(pl.col("week") == POISON_WEEK).row(0, named=True)

    completed = matrix.filter(pl.col("gameday") < bound["kickoff"])
    train, calib = split_windows(completed)

    assert calib["season"].n_unique() == CALIB_WINDOW_SEASONS
    assert calib["season"].max() == SAMPLE_SEASON, "the season in progress calibrates"
    assert train["season"].max() < calib["season"].min()
    assert train.height + calib.height == completed.height


def test_every_scheduled_game_is_predicted_exactly_once(matrix, season_walk):
    """A walk that silently dropped or duplicated a week would still 'work'."""
    played = matrix.filter(pl.col("season") == SAMPLE_SEASON)

    assert season_walk.predictions.height == played.height
    assert (
        season_walk.predictions["game_id"].n_unique() == season_walk.predictions.height
    )
    assert set(season_walk.predictions["game_id"]) == set(played["game_id"])


def test_probabilities_are_in_range(season_walk):
    for column in ("ensemble", "elo_raw", "elo_calibrated"):
        values = season_walk.predictions[column].to_numpy()
        assert ((values > 0.0) & (values < 1.0)).all(), column


# --------------------------------------------------------- runtime poisoning


def _poison(matrix: pl.DataFrame, league_week: int) -> pl.DataFrame:
    """Rewrite every feature and result at or after ``league_week`` to garbage."""
    at_or_after = pl.col("league_week") >= league_week
    return matrix.with_columns(
        [
            pl.when(at_or_after).then(999.0).otherwise(pl.col(c)).alias(c)
            for c in CORE_FEATURES
        ]
        + [
            pl.when(at_or_after).then(999.0).otherwise(pl.col("elo_prob")).alias("elo_prob"),
            pl.when(at_or_after).then(1.0).otherwise(pl.col("home_win")).alias("home_win"),
        ]
    )


@pytest.fixture(scope="module")
def poison_target(matrix: pl.DataFrame) -> dict:
    boundaries = week_boundaries(matrix, (SAMPLE_SEASON, SAMPLE_SEASON))
    return boundaries.filter(pl.col("week") == POISON_WEEK).row(0, named=True)


def _predict_target(source: pl.DataFrame, clean_target: pl.DataFrame, bound: dict):
    """Fit from ``source``, score the **clean** target week.

    The scoring frame is deliberately the untouched one, exactly as
    ``test_models.py::test_fitting_ignores_the_test_split`` scores validation
    after poisoning test: poisoning the rows a model is *asked about* would move
    its answers for the obvious reason and tell us nothing about its fit.
    """
    completed = source.filter(pl.col("gameday") < bound["kickoff"]).sort(
        "gameday", "game_id"
    )
    fit = fit_week(
        completed,
        int(bound["league_week"]),
        SAMPLE_SEASON,
        POISON_WEEK,
        ENSEMBLE_FEATURES,
        HALF_LIFE,
    )
    return score_week(fit, clean_target)["ensemble"].to_numpy()


def test_poisoning_the_target_week_and_after_changes_nothing(matrix, poison_target):
    """D-29's replacement for the AST guard.

    Every feature and every result from the target week onward becomes garbage.
    A harness that reached forward — by a join, a global mean, a scaler fitted
    on the whole matrix, or a week cut by number rather than by kickoff — would
    move. This one must not move at all, to the last bit.
    """
    league_week = int(poison_target["league_week"])
    clean_target = matrix.filter(
        (pl.col("season") == SAMPLE_SEASON) & (pl.col("week") == POISON_WEEK)
    ).sort("gameday", "game_id")

    poisoned = _poison(matrix, league_week)
    assert poisoned.filter(pl.col("league_week") >= league_week).height, "nothing poisoned"

    np.testing.assert_array_equal(
        _predict_target(matrix, clean_target, poison_target),
        _predict_target(poisoned, clean_target, poison_target),
        err_msg=(
            "the target week's predictions moved when games at and after it were "
            "rewritten — the harness is fitting on games that had not kicked off"
        ),
    )


def test_poisoning_before_the_target_week_does_change_things(matrix, poison_target):
    """Guards the test above from passing vacuously.

    Poisons the *earlier* rows instead — the ones the harness is supposed to be
    fitting on. Features rather than results, so both classes survive; and only
    the 20 league weeks immediately before the target, because rewriting every
    completed row would leave every column constant and the estimators would
    fail to fit rather than fit differently.
    """
    league_week = int(poison_target["league_week"])
    clean_target = matrix.filter(
        (pl.col("season") == SAMPLE_SEASON) & (pl.col("week") == POISON_WEEK)
    ).sort("gameday", "game_id")

    before = pl.col("league_week").is_between(league_week - 20, league_week - 1)
    earlier = matrix.with_columns(
        [
            pl.when(before).then(999.0).otherwise(pl.col(c)).alias(c)
            for c in CORE_FEATURES
        ]
    )

    assert not np.allclose(
        _predict_target(matrix, clean_target, poison_target),
        _predict_target(earlier, clean_target, poison_target),
    )


# ------------------------------------------------------------- the weighting


def test_the_weight_column_reaches_the_fit(matrix, poison_target):
    """The recency column is attached to both windows, not just the base fit."""
    kickoff = poison_target["kickoff"]
    completed = matrix.filter(pl.col("gameday") < kickoff)
    fit = fit_week(
        completed,
        int(poison_target["league_week"]),
        SAMPLE_SEASON,
        POISON_WEEK,
        ENSEMBLE_FEATURES,
        HALF_LIFE,
    )

    for frame in (fit.train, fit.calib):
        assert WEIGHT_COLUMN in frame.columns
        weights = frame[WEIGHT_COLUMN].to_numpy()
        assert ((weights > 0.0) & (weights <= 1.0)).all()

    # The calibration window is the recent one, so it must weigh more per game.
    assert fit.calib[WEIGHT_COLUMN].mean() > fit.train[WEIGHT_COLUMN].mean()


def test_with_recency_leaves_the_frame_otherwise_untouched(matrix):
    sample = matrix.head(50)
    weighted = with_recency(sample, 400, 20.0)

    assert weighted.columns == [*sample.columns, WEIGHT_COLUMN]
    assert weighted.drop(WEIGHT_COLUMN).equals(sample)
