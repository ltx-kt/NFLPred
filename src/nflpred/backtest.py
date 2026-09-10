"""Walk-forward retraining, and the recency weighting that rides with it.

Phases 1-6 are a **static** experiment: one base fit on 2006-2015, one
calibrator on 2016-2018, scored on 2019-2021 and never refitted. That answers
"how good is this model" and not "how good is this system", which are different
questions once a season is in progress and the answer is supposed to arrive on a
Tuesday. This module is the second question.

**What moves, and what does not (D-26).** Both windows roll forward. For a
target week, the base estimators are fitted on everything completed before its
earliest kickoff *except* the trailing :data:`~nflpred.config.CALIB_WINDOW_SEASONS`
seasons, which the calibrator gets and the base fit never sees. That is D-5's
structure - bases and calibrators never share rows - moved through time rather
than abandoned. Everything else about the fit is the D-22 construction,
unchanged: five estimators from the factory, Platt-scaled Elo, prefit stack A.

**Why recency weighting is here and not in Phase 4.** D-18 found the calibrator
had been fitted to a 58.3%-home-win era and scored in a 51.3% one, and made log
loss *worse* for it. The spec named recency weighting as the fix and explicitly
ruled out the alternative, which was tuning the calibrator against validation.
Weights ride D-4's ``sample_weight`` channel, multiplying into the tie encoding
rather than replacing it - :func:`~nflpred.evaluate.expand_ties` composes them, and
`tests/test_recency.py` exercises the composition rather than assuming it.

**The one thing this module must never do** is fit on a game that had not
kicked off. Every window is cut by ``gameday`` against the target week's
earliest kickoff, never by week number: a Thursday game and the Monday game
after it are the same "week" and eleven days apart, and week numbers are not
comparable across seasons at all. `tests/test_walk_forward.py` checks the
chronology directly and then poisons everything at or after the target week to
check it behaviourally.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import polars as pl
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression

from nflpred.config import CALIB_WINDOW_SEASONS, RECENCY_HALF_LIFE
from nflpred.features.build import CORE_FEATURES
from nflpred.modeling.base import (
    build_models,
    elo_platt_probability,
    elo_probability,
    fit_all,
    fit_calibrated,
    fit_elo_platt,
    home_win_probability,
    inner_estimator,
    split_frame,
)
from nflpred.modeling.ensemble import (
    ENSEMBLE_FEATURES,
    MEMBER_ORDER,
    Member,
    build_members,
    member_probabilities,
    prefit_stack,
)

#: The column :func:`walk_forward` writes its weights into and hands to
#: ``to_training_arrays(..., weight=)``. One name, so a fit that silently lost
#: the weighting would be a missing column rather than a plausible number.
WEIGHT_COLUMN: str = "recency"

#: The column :func:`recency_weights` measures lag against. League weeks, not
#: calendar weeks - see :func:`~nflpred.features.build.league_week_index`.
CLOCK_COLUMN: str = "league_week"


def recency_weights(
    frame: pl.DataFrame, target_week_index: int, half_life: float | None
) -> pl.Series:
    """``0.5 ** (weeks_ago / half_life)``, one weight per row of ``frame``.

    The spec's decay, verbatim, on the league-week clock. A game in the target
    week itself would weigh 1.0; one half-life back weighs 0.5; the decay never
    reaches zero, so an old game is downweighted rather than discarded.

    ``half_life=None`` returns all ones. That is deliberately the *same code
    path* rather than a branch the caller takes: the unweighted control has to
    be the identical fit with a flat weight vector, or the grid's ``None`` cell
    would be measuring the presence of a ``sample_weight`` argument as well as
    the weighting itself.
    """
    if half_life is None:
        return pl.Series(WEIGHT_COLUMN, np.ones(frame.height, dtype=float))

    lag = float(target_week_index) - frame[CLOCK_COLUMN].to_numpy().astype(float)
    return pl.Series(WEIGHT_COLUMN, 0.5 ** (lag / float(half_life)))


def with_recency(
    frame: pl.DataFrame, target_week_index: int, half_life: float | None
) -> pl.DataFrame:
    """``frame`` with :data:`WEIGHT_COLUMN` attached."""
    return frame.with_columns(recency_weights(frame, target_week_index, half_life))


# ------------------------------------------------------------- the weekly fit


@dataclass(frozen=True)
class WeeklyFit:
    """Everything one target week's refit produced, kept in memory.

    Held rather than pickled: D-30 leaves native model formats deferred, and
    this harness refits per week and never reloads across environments. The
    objects here are what `nflpred.predict` explains and logs, and the frames are
    what its config row records.
    """

    season: int
    week: int
    league_week: int
    models: dict[str, ClassifierMixin]
    raw: dict[str, ClassifierMixin]
    platt: LogisticRegression
    stack: ClassifierMixin
    members: list[Member]
    raw_members: list[Member]
    train: pl.DataFrame
    calib: pl.DataFrame
    half_life: float | None
    features: Sequence[str]
    seconds: float

    def seasons(self, which: str) -> tuple[int, int]:
        """``(first, last)`` season of the ``train`` or ``calib`` window."""
        frame = self.train if which == "train" else self.calib
        return int(frame["season"].min()), int(frame["season"].max())

    @property
    def last_fitted_kickoff(self) -> date:
        """The most recent game either window saw.

        Reported per week so the chronology claim is a column of the harness's
        own output rather than something a test has to re-derive from the
        matrix. `tests/test_walk_forward.py` asserts it is strictly before every
        target week's first kickoff.
        """
        return max(self.train["gameday"].max(), self.calib["gameday"].max())


def split_windows(
    completed: pl.DataFrame, calib_seasons: int = CALIB_WINDOW_SEASONS
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Cut ``completed`` into a base-fit window and a calibration window (D-26).

    The calibration window is the last ``calib_seasons`` **seasons present in
    ``completed``**, which mid-season includes the season in progress: for a
    target of 2019 week 5, the calibrator gets 2017, 2018 and the four completed
    weeks of 2019, and the base fit gets everything through 2016. At a week-1
    target the two windows land exactly on D-5's frozen boundaries, which is the
    sense in which this generalises that decision rather than replacing it.

    The alternative reading - only seasons that have *finished* - would push the
    in-progress season into the base fit, putting the freshest games behind the
    calibrator rather than in front of it. That is the wrong side for the one
    thing recency weighting exists to fix (D-18).
    """
    seasons = sorted(completed["season"].unique().to_list())
    if len(seasons) <= calib_seasons:
        msg = (
            f"only {len(seasons)} completed seasons ({seasons}) before this week, "
            f"which the {calib_seasons}-season calibration window would consume "
            f"entirely, leaving nothing to fit the bases on. The matrix starts at "
            f"2006 (D-2), so the earliest walkable target is season "
            f"{seasons[0] + calib_seasons}."
        )
        raise ValueError(msg)

    boundary = seasons[-calib_seasons]
    calib = completed.filter(pl.col("season") >= boundary)
    train = completed.filter(pl.col("season") < boundary)
    return train, calib


def fit_week(
    completed: pl.DataFrame,
    league_week: int,
    season: int,
    week: int,
    features: Sequence[str] = ENSEMBLE_FEATURES,
    half_life: float | None = RECENCY_HALF_LIFE,
    calib_seasons: int = CALIB_WINDOW_SEASONS,
) -> WeeklyFit:
    """The D-22 construction, fitted on one week's worth of history.

    ``completed`` must already exclude the target week. This function does not
    check that - :func:`walk_forward` and :mod:`nflpred.predict` both cut the frame
    by kickoff before calling, and re-deriving the cut here from a week number
    would introduce the second definition the module docstring warns about.

    ``features`` is the 17-column ensemble tuple and is what the *stack* is
    fitted on. The five members are always fitted on
    :data:`~nflpred.features.build.CORE_FEATURES`, because
    :data:`~nflpred.modeling.ensemble.BASE_COLUMNS` hard-codes the slice they read
    back - that pairing is the ensemble module's layout, not a choice here.
    """
    started = time.perf_counter()

    train, calib = split_windows(completed, calib_seasons)
    train = with_recency(train, league_week, half_life)
    calib = with_recency(calib, league_week, half_life)

    models = {
        name: fit_calibrated(
            estimator, train, calib, CORE_FEATURES, weight=WEIGHT_COLUMN
        )
        for name, estimator in build_models(CORE_FEATURES).items()
    }
    raw = {name: inner_estimator(model) for name, model in models.items()}
    platt = fit_elo_platt(calib, weight=WEIGHT_COLUMN)

    raw_members = build_members(raw)
    members = build_members(models, platt=platt)
    stack = prefit_stack(raw_members, calib, features, weight=WEIGHT_COLUMN)

    return WeeklyFit(
        season=season,
        week=week,
        league_week=league_week,
        models=models,
        raw=raw,
        platt=platt,
        stack=stack,
        members=members,
        raw_members=raw_members,
        train=train,
        calib=calib,
        half_life=half_life,
        features=tuple(features),
        seconds=time.perf_counter() - started,
    )


def score_week(fit: WeeklyFit, target: pl.DataFrame) -> pl.DataFrame:
    """One row per game of ``target``: the stack, every member, and the raw Elo.

    Members are the **calibrated** six - the votes the output contract reports
    (D-21's note) - while the stack holds the raw ones, which is stack A's whole
    construction. Both are emitted flat so that the per-member rolling Brier the
    spec asks for is a query rather than a JSON walk.
    """
    features = list(fit.features)
    probability = home_win_probability(fit.stack, target, features)
    per_member = member_probabilities(fit.members, target, features)

    return target.select(
        "game_id",
        "season",
        "week",
        "league_week",
        "gameday",
        "home_team",
        "away_team",
        "home_win",
        "home_moneyline",
        "away_moneyline",
    ).with_columns(
        fit_season=pl.lit(fit.season, dtype=pl.Int32),
        fit_week=pl.lit(fit.week, dtype=pl.Int32),
        ensemble=pl.Series("ensemble", probability),
        elo_raw=pl.Series("elo_raw", elo_probability(target)),
        elo_calibrated=pl.Series(
            "elo_calibrated", elo_platt_probability(fit.platt, elo_probability(target))
        ),
        **{f"member_{name}": pl.Series(per_member[name]) for name in MEMBER_ORDER},
    )


# ---------------------------------------------------------------- the harness


@dataclass(frozen=True)
class WalkForward:
    """What a backtest produced: the per-game scores, and how it got there."""

    predictions: pl.DataFrame
    weeks: pl.DataFrame
    half_life: float | None
    features: tuple[str, ...]

    @property
    def y(self) -> np.ndarray:
        return self.predictions["home_win"].to_numpy().astype(float)

    def probability(self, column: str = "ensemble") -> np.ndarray:
        return self.predictions[column].to_numpy().astype(float)


def week_boundaries(matrix: pl.DataFrame, seasons: tuple[int, int]) -> pl.DataFrame:
    """Every ``(season, week)`` in ``seasons``, with its earliest kickoff.

    Ordered by that kickoff rather than by ``(season, week)``. They agree on
    real schedules, and ordering by the thing the windows are actually cut with
    means they cannot disagree here without the disagreement being visible.
    """
    return (
        matrix.filter(pl.col("season").is_between(*seasons))
        .group_by("season", "week")
        .agg(
            kickoff=pl.col("gameday").min(),
            league_week=pl.col("league_week").first(),
            games=pl.len(),
        )
        .sort("kickoff", "season", "week")
    )


def walk_forward(
    matrix: pl.DataFrame,
    seasons: tuple[int, int],
    features: Sequence[str] = ENSEMBLE_FEATURES,
    half_life: float | None = RECENCY_HALF_LIFE,
    calib_seasons: int = CALIB_WINDOW_SEASONS,
    cadence: int = 1,
    on_week: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> WalkForward:
    """Refit before every week of ``seasons``, predict it, and accumulate.

    ``cadence`` is how many target weeks one fit serves. It is a parameter
    because a weekly refit of ten models is the dominant cost of both this
    harness and the tuning grid, and **production and tuning must use the same
    value** - a half-life gridded at cadence 4 and served at cadence 1 was
    chosen for a different system. The default is 1 and nothing in the project
    changes it.

    ``on_week`` is called after each target week with
    ``(index, total, summary)`` so a long run can report progress without this
    module owning a print format.
    """
    boundaries = week_boundaries(matrix, seasons)
    total = boundaries.height

    scored: list[pl.DataFrame] = []
    rows: list[dict[str, Any]] = []
    fit: WeeklyFit | None = None

    for index, bound in enumerate(boundaries.iter_rows(named=True)):
        season, week, kickoff = bound["season"], bound["week"], bound["kickoff"]

        target = matrix.filter(
            (pl.col("season") == season) & (pl.col("week") == week)
        ).sort("gameday", "game_id")
        completed = matrix.filter(pl.col("gameday") < kickoff).sort("gameday", "game_id")

        if fit is None or index % cadence == 0:
            fit = fit_week(
                completed,
                int(bound["league_week"]),
                season,
                week,
                features,
                half_life,
                calib_seasons,
            )

        scored.append(score_week(fit, target))

        summary = {
            "season": season,
            "week": week,
            "league_week": int(bound["league_week"]),
            "games": target.height,
            "refit": fit.season == season and fit.week == week,
            "kickoff": kickoff,
            "last_fitted_kickoff": fit.last_fitted_kickoff,
            "n_train": fit.train.height,
            "n_calib": fit.calib.height,
            "train_seasons": "-".join(str(s) for s in fit.seasons("train")),
            "calib_seasons": "-".join(str(s) for s in fit.seasons("calib")),
            "fit_seconds": round(fit.seconds, 2),
        }
        rows.append(summary)
        if on_week is not None:
            on_week(index + 1, total, summary)

    return WalkForward(
        predictions=pl.concat(scored, how="vertical").sort("gameday", "game_id"),
        weeks=pl.DataFrame(rows),
        half_life=half_life,
        features=tuple(features),
    )


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish's ``(sum w)^2 / sum w^2`` - how many games a weighted fit is worth.

    Reported rather than assumed, because it is the whole cost side of the
    recency trade: a half-life short enough to track a shifting home-field
    advantage is also short enough to throw away most of the 2,670 games the
    frozen split had. If the grid's ``None`` cell wins, this number is why.
    """
    w = np.asarray(weights, dtype=float)
    return float(w.sum() ** 2 / np.square(w).sum())


def frozen_split_reference(
    matrix: pl.DataFrame,
    target: pl.DataFrame,
    features: Sequence[str] = ENSEMBLE_FEATURES,
) -> np.ndarray:
    """The Phase 5 stack, fitted on the frozen split, scored on ``target``.

    D-26 retires the carried constraint that every phase trains on 2006-2015,
    which would otherwise make the walk-forward number incomparable with every
    number in the README. This puts the old construction and the new one on the
    same games so the comparison is a row of the table rather than an argument.

    Reads only the ``train`` and ``calib`` splits, like Phase 5 did.
    """
    calib = split_frame(matrix, "calib")
    models = fit_all(matrix, CORE_FEATURES)
    raw = {name: inner_estimator(model) for name, model in models.items()}
    stack = prefit_stack(build_members(raw), calib, features)
    return home_win_probability(stack, target, features)
