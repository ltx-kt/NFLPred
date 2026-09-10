"""The live path, and the one test the whole phase rests on.

Take a **completed** week. Strip it back to what was knowable before kickoff -
no scores, no play-by-play, no box score - synthesize it exactly as
:func:`nflpred.predict.week_frame` synthesizes a 2026 fixture, and demand that its
``ENSEMBLE_FEATURES`` row come back **identical** to the row the real matrix
holds for the same game.

That is train/serve skew measured rather than argued, and it is the whole
justification for D-8's prior-starter proxy. If it fails, the live path is
predicting on different numbers than it trained on and nothing downstream is
trustworthy - every calibration curve, every SHAP attribution and every
confidence band would be describing a model being fed inputs it never saw.

It is not a formality. This test is what found D-31: the quarterback composite
resolved a projected starter's form by an as-of join at his own start, which is
a row an unplayed fixture does not have, so all three ``qb_*`` columns came back
one start stale - on every game, in exactly the direction that would never show
up in a backtest.
"""

from __future__ import annotations

import polars as pl
import pytest

from nflpred.config import LIVE_SEASON, TEAM_GAME_GT_PATH
from nflpred.features.build import build_features
from nflpred.ingest import read_pbp, read_schedules
from nflpred.modeling.ensemble import ENSEMBLE_FEATURES
from nflpred.predict import _synthesized_rows, week_frame

pytestmark = pytest.mark.data

#: A completed week, mid-season so every team has a full window behind it and
#: some of them have changed quarterback. Fixed rather than random so a failure
#: is reproducible.
SKEW_SEASON: int = 2019
SKEW_WEEK: int = 8

#: A week of the live season. Its schedule exists; nothing else about it does.
LIVE_WEEK: int = 1


@pytest.fixture(scope="module")
def sources() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    if not TEAM_GAME_GT_PATH.exists():
        pytest.skip(
            "team_game_gt.parquet not built; "
            "run python -m nflpred.features.pbp_agg --garbage-time"
        )
    return pl.read_parquet(TEAM_GAME_GT_PATH), read_schedules(), read_pbp()


@pytest.fixture(scope="module")
def as_if_unplayed(sources) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """The real matrix, and one rebuilt with a completed week hidden.

    Everything the week produced is removed from every source: the scores from
    the schedule, the plays from the play-by-play, the aggregates from the
    team-game table. What is left is what the system would have had on the
    Thursday morning - and the synthesized rows are built the way
    `nflpred.predict.week_frame` builds them, from the same helper.
    """
    team_game, schedules, pbp = sources

    week = schedules.filter(
        (pl.col("season") == SKEW_SEASON) & (pl.col("week") == SKEW_WEEK)
    )
    game_ids = week["game_id"].to_list()
    assert game_ids, f"{SKEW_SEASON} week {SKEW_WEEK} is not in the schedule"

    hidden_schedules = schedules.with_columns(
        [
            pl.when(pl.col("game_id").is_in(game_ids))
            .then(pl.lit(None, dtype=schedules.schema[c]))
            .otherwise(pl.col(c))
            .alias(c)
            for c in ("home_score", "away_score", "result")
        ]
    )
    hidden_pbp = pbp.filter(~pl.col("game_id").is_in(game_ids))
    hidden_team_game = pl.concat(
        [
            team_game.filter(~pl.col("game_id").is_in(game_ids)),
            _synthesized_rows(week, team_game),
        ],
        how="vertical",
    )

    return (
        build_features(team_game, schedules, pbp),
        build_features(hidden_team_game, hidden_schedules, hidden_pbp),
        game_ids,
    )


def _rows(matrix: pl.DataFrame, game_ids: list[str]) -> pl.DataFrame:
    return matrix.filter(pl.col("game_id").is_in(game_ids)).sort("game_id")


# ----------------------------------------------------- the one that matters


def test_a_synthesized_week_matches_the_real_matrix_feature_for_feature(as_if_unplayed):
    """No train/serve skew: the same game, built two ways, is the same row."""
    real, synthesized, game_ids = as_if_unplayed
    before, after = _rows(real, game_ids), _rows(synthesized, game_ids)

    assert before.height == len(game_ids)
    assert after.height == len(game_ids)

    for feature in ENSEMBLE_FEATURES:
        assert before[feature].to_list() == after[feature].to_list(), (
            f"{feature} differs between the played and the pre-kickoff build - "
            f"the live path would predict on numbers the model never trained on"
        )


def test_the_synthesized_week_really_has_no_result(as_if_unplayed):
    """Guards the test above: the week must genuinely have been hidden."""
    _, synthesized, game_ids = as_if_unplayed
    rows = _rows(synthesized, game_ids)

    assert rows["home_win"].null_count() == rows.height
    assert rows["home_moneyline"].null_count() < rows.height, (
        "the market line is a pre-kickoff quantity and must survive the hiding"
    )


def test_hiding_a_week_does_not_disturb_earlier_rows(as_if_unplayed):
    """The synthesized row is last in its team's series, so nothing before moves."""
    real, synthesized, game_ids = as_if_unplayed
    kickoff = _rows(real, game_ids)["gameday"].min()

    before = real.filter(pl.col("gameday") < kickoff).sort("game_id")
    after = synthesized.filter(pl.col("gameday") < kickoff).sort("game_id")

    assert before.height == after.height
    assert before.select(ENSEMBLE_FEATURES).equals(after.select(ENSEMBLE_FEATURES))


def test_no_feature_is_null_on_a_synthesized_row(as_if_unplayed):
    """The claim `week_frame` asserts at runtime, checked here on real data.

    Every rolling expression is shift(1)ed and there is exactly one unplayed row
    per team, so no null can enter a window. Asserted rather than assumed (D-27).
    """
    _, synthesized, game_ids = as_if_unplayed
    rows = _rows(synthesized, game_ids)

    for feature in ENSEMBLE_FEATURES:
        assert rows[feature].null_count() == 0, feature


# ------------------------------------------------------------ week_frame


def test_week_frame_on_a_completed_week_is_a_slice_of_the_matrix(sources):
    team_game, schedules, pbp = sources
    view = week_frame(SKEW_SEASON, SKEW_WEEK, team_game, schedules, pbp)
    real = build_features(team_game, schedules, pbp)

    assert not view.live
    assert not view.synthesized
    assert view.target.equals(_rows(real, view.target["game_id"].to_list()).sort(
        "gameday", "game_id"
    ))


def test_week_frame_fits_only_on_games_that_kicked_off_earlier(sources):
    team_game, schedules, pbp = sources
    view = week_frame(SKEW_SEASON, SKEW_WEEK, team_game, schedules, pbp)

    assert view.completed["gameday"].max() < view.kickoff
    assert view.completed["home_win"].null_count() == 0
    assert not set(view.completed["game_id"]) & set(view.target["game_id"])


def test_week_frame_builds_the_live_season(sources):
    """The genuine live path: a season with a schedule and no results at all."""
    team_game, schedules, pbp = sources
    if not schedules.filter(pl.col("season") == LIVE_SEASON).height:
        pytest.skip(f"{LIVE_SEASON} is not in the cached schedule yet")

    view = week_frame(LIVE_SEASON, LIVE_WEEK, team_game, schedules, pbp)

    assert view.live
    assert len(view.synthesized) == view.target.height
    assert view.target["home_win"].null_count() == view.target.height
    for feature in ENSEMBLE_FEATURES:
        assert view.target[feature].null_count() == 0, feature
    # Nothing from the live season could have been fitted on: there is nothing.
    assert view.completed["season"].max() < LIVE_SEASON


def test_week_frame_refuses_a_week_that_is_not_next(sources):
    """The documented limit, as a refusal rather than a silently wrong answer."""
    team_game, schedules, pbp = sources
    if not schedules.filter(pl.col("season") == LIVE_SEASON).height:
        pytest.skip(f"{LIVE_SEASON} is not in the cached schedule yet")

    with pytest.raises(ValueError, match="not the next unplayed week"):
        week_frame(LIVE_SEASON, 8, team_game, schedules, pbp)


def test_week_frame_rejects_a_week_that_does_not_exist(sources):
    team_game, schedules, pbp = sources
    with pytest.raises(ValueError, match="no games scheduled"):
        week_frame(SKEW_SEASON, 99, team_game, schedules, pbp)


# ------------------------------------------------------------ synthesis


def test_synthesized_rows_are_two_per_game_and_carry_no_statistics(sources):
    team_game, schedules, _ = sources
    week = schedules.filter(
        (pl.col("season") == SKEW_SEASON) & (pl.col("week") == SKEW_WEEK)
    )
    rows = _synthesized_rows(week, team_game)

    assert rows.height == 2 * week.height
    assert rows.columns == team_game.columns
    assert rows.schema == team_game.schema
    assert rows["won"].null_count() == rows.height
    assert rows["is_home"].sum() == week.height

    for column in ("off_epa_per_play", "def_epa_per_play", "points_for", "margin"):
        assert rows[column].null_count() == rows.height, column
