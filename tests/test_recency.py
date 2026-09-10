"""Recency weighting: the decay, and the composition D-14 assumed.

Three claims, and the third is the one that has never been exercised anywhere
else in the project.

**The decay is the spec's decay.** ``0.5 ** (weeks_ago / half_life)`` on the
league-week clock: 1.0 in the target week, exactly 0.5 one half-life back,
strictly decreasing, never zero.

**``None`` is the same code path.** The unweighted control has to be the
identical fit with a flat weight vector, not a fit that skipped the
``sample_weight`` argument — otherwise the tuning grid's ``None`` cell would be
measuring two differences at once. Pinned by comparing predictions bit for bit.

**Weights compose by multiplication.** D-4 encodes a tie as two half-weight
rows and D-14 confirmed it; both took for granted that an outer weight would
multiply cleanly through :func:`~nflpred.evaluate.expand_ties`. Phase 7 is the first
thing to actually pass an outer weight, so a tie inside a recency window is
checked here rather than assumed — a tie at recency *r* must carry ``0.5 * r``,
twice, and contribute exactly *r* of total weight.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nflpred.backtest import (
    CLOCK_COLUMN,
    WEIGHT_COLUMN,
    effective_sample_size,
    recency_weights,
    with_recency,
)
from nflpred.features.build import CORE_FEATURES, to_training_arrays
from nflpred.modeling.base import (
    build_models,
    fit_calibrated,
    home_win_probability,
)

HALF_LIFE: float = 20.0


pytestmark = pytest.mark.data

def _clock(values: list[int]) -> pl.DataFrame:
    return pl.DataFrame({CLOCK_COLUMN: pl.Series(values, dtype=pl.Int32)})


# ------------------------------------------------------------------ the decay


def test_weight_is_one_at_zero_lag():
    """A game in the target week itself is at full weight."""
    weights = recency_weights(_clock([100]), 100, HALF_LIFE)
    assert weights[0] == pytest.approx(1.0)


def test_weight_halves_every_half_life():
    frame = _clock([100, 80, 60, 40])
    weights = recency_weights(frame, 100, HALF_LIFE).to_list()

    assert weights == pytest.approx([1.0, 0.5, 0.25, 0.125])


def test_weights_are_strictly_decreasing_with_age():
    frame = _clock(list(range(1, 200)))
    weights = recency_weights(frame, 200, HALF_LIFE).to_numpy()

    assert (np.diff(weights) > 0).all(), "older games must weigh less, without ties"
    assert (weights > 0).all(), "the decay downweights, it never discards"


def test_none_half_life_is_all_ones():
    frame = _clock([1, 50, 400])
    weights = recency_weights(frame, 400, None)

    assert weights.to_list() == [1.0, 1.0, 1.0]
    assert weights.name == WEIGHT_COLUMN


def test_matrix_league_week_is_a_dense_chronological_index(matrix):
    """The clock the decay runs on: one step per week *of football*."""
    weeks = matrix.select("season", "week", CLOCK_COLUMN).unique().sort(CLOCK_COLUMN)

    assert weeks[CLOCK_COLUMN].to_list() == list(range(1, weeks.height + 1))
    # Chronological: sorting by the index sorts by the calendar.
    assert weeks.select("season", "week").equals(
        weeks.select("season", "week").sort("season", "week")
    )
    # And the offseason is not counted — the last week of a season and the first
    # of the next are one step apart, not thirty.
    boundaries = weeks.filter(pl.col("week") == 1)
    assert boundaries.height == matrix["season"].n_unique()


# ----------------------------------------------------- None is the same path


def test_unweighted_half_life_reproduces_the_unweighted_fit_exactly(splits):
    """``half_life=None`` must be the *same* fit, not merely a similar one."""
    train, calib, val = splits

    plain = fit_calibrated(
        build_models(CORE_FEATURES)["logreg"], train, calib, CORE_FEATURES
    )
    flat = fit_calibrated(
        build_models(CORE_FEATURES)["logreg"],
        with_recency(train, 400, None),
        with_recency(calib, 400, None),
        CORE_FEATURES,
        weight=WEIGHT_COLUMN,
    )

    np.testing.assert_array_equal(
        home_win_probability(plain, val, CORE_FEATURES),
        home_win_probability(flat, val, CORE_FEATURES),
    )


def test_a_real_half_life_moves_the_fit(splits):
    """Guards the test above from passing vacuously."""
    train, calib, val = splits
    target = int(train[CLOCK_COLUMN].max()) + 1

    flat = fit_calibrated(
        build_models(CORE_FEATURES)["logreg"],
        with_recency(train, target, None),
        with_recency(calib, target, None),
        CORE_FEATURES,
        weight=WEIGHT_COLUMN,
    )
    decayed = fit_calibrated(
        build_models(CORE_FEATURES)["logreg"],
        with_recency(train, target, HALF_LIFE),
        with_recency(calib, target, HALF_LIFE),
        CORE_FEATURES,
        weight=WEIGHT_COLUMN,
    )

    assert not np.allclose(
        home_win_probability(flat, val, CORE_FEATURES),
        home_win_probability(decayed, val, CORE_FEATURES),
    )


# ------------------------------------------------------------- composition


def test_a_tie_carries_half_the_recency_weight(splits):
    """D-14's stated assumption, exercised for the first time.

    A tie is two rows at half weight (D-4). Under recency weighting those two
    rows must carry ``0.5 * r`` each, not 0.5 and not *r* — the two mechanisms
    multiply, and if they did not, a tie in a recent week would count as two
    whole games or a recent game would count as an old one.
    """
    train, _, _ = splits
    ties = train.filter(pl.col("home_win") == 0.5)
    assert ties.height, "the training split should contain at least one tie"

    target = int(train[CLOCK_COLUMN].max()) + 1
    weighted = with_recency(ties, target, HALF_LIFE)
    expected = recency_weights(ties, target, HALF_LIFE).to_numpy()

    _, y, w = to_training_arrays(weighted, CORE_FEATURES, weight=WEIGHT_COLUMN)

    assert len(w) == 2 * ties.height
    np.testing.assert_allclose(w, np.concatenate([expected * 0.5, expected * 0.5]))
    assert sorted(y.tolist()) == [0.0] * ties.height + [1.0] * ties.height
    # Each tie still contributes exactly its own recency weight in total.
    assert w.sum() == pytest.approx(expected.sum())


def test_a_decided_game_carries_its_recency_weight_undivided(splits):
    """The other half of the composition: no halving where there is no tie."""
    train, _, _ = splits
    decided = train.filter(pl.col("home_win") != 0.5).head(200)

    target = int(train[CLOCK_COLUMN].max()) + 1
    expected = recency_weights(decided, target, HALF_LIFE).to_numpy()
    _, _, w = to_training_arrays(
        with_recency(decided, target, HALF_LIFE), CORE_FEATURES, weight=WEIGHT_COLUMN
    )

    np.testing.assert_allclose(w, expected)


# ---------------------------------------------------------------- the cost


def test_effective_sample_size_falls_as_the_half_life_shortens(splits):
    """The cost side of the trade, as a number rather than an intuition."""
    train, _, _ = splits
    target = int(train[CLOCK_COLUMN].max()) + 1

    sizes = [
        effective_sample_size(recency_weights(train, target, hl).to_numpy())
        for hl in (None, 60.0, 40.0, 20.0, 10.0)
    ]

    assert sizes[0] == pytest.approx(float(train.height))
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[-1] < train.height / 2, (
        "a 10-league-week half-life should cost most of the sample; if it does "
        "not, the clock is not measuring what the docstring says it measures"
    )
