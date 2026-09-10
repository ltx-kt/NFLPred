"""Constraint 1, in a form that is actually checkable.

The spec asks for "an explicit assertion that verifies no feature column was
derived from the target game's ``game_id``". Provenance cannot be read off a
dataframe — by the time a value is a float, where it came from is gone. So
these tests are behavioural instead, and between them they pin the property
down tighter than a provenance check would:

* **Perturbation** — rewrite one game's box score, rebuild, and the same game's
  own feature row must be unchanged. If any feature had touched the target
  game, this fails. The strongest single check here.
* **Recomputation** — re-derive the rolling value from strictly prior games by
  a completely separate code path and demand equality.
* **Chronology** — every contributing game kicked off before the target did.
* **Correlation tripwire** — no feature correlates with the target above 0.9,
  which is what a target-derived column sneaking in would look like.

These run against the real built tables, not synthetic ones. A leak that only
appears at the seam between two seasons or two franchises would never show up
in a toy frame.

From Phase 3 the perturbation runs the **whole** pipeline — rolling form, Elo
and the QB composite — and rewrites the game's *result* as well as its box
score. Rolling form is blind to who won, so a box-score-only perturbation would
have exercised none of Elo; a result-only one would have exercised none of the
form. The Elo and QB modules also have their own perturbation tests in
``test_elo.py`` and ``test_qb.py``, which localise a failure this one would only
report as "something moved".
"""

from __future__ import annotations

import polars as pl
import pytest

from nflpred.config import (
    FEATURE_MATRIX_PATH,
    LAST_COMPLETED_SEASON,
    MATRIX_START_SEASON,
    ROLLING_WINDOWS,
    TEAM_GAME_PATH,
)
from nflpred.features.build import (
    BASELINE_FEATURES,
    PHASE3_FEATURES,
    build_features,
    net_feature_names,
    wide_features,
)
from nflpred.features.rolling import build_rolling
from nflpred.ingest import read_pbp, read_schedules

pytestmark = pytest.mark.data

#: Fixed so a failure is reproducible; the sample is arbitrary, not random per run.
SAMPLE_SEED: int = 20260814
SAMPLE_SIZE: int = 50

#: The window every 2006+ row is guaranteed to have behind it.
LONGEST_WINDOW: int = max(ROLLING_WINDOWS)


@pytest.fixture(scope="module")
def team_game() -> pl.DataFrame:
    if not TEAM_GAME_PATH.exists():
        pytest.skip("team_game.parquet not built; run python -m nflpred.features.pbp_agg")
    return pl.read_parquet(TEAM_GAME_PATH).with_columns(pl.col("gameday").str.to_date())


@pytest.fixture(scope="module")
def rolling(team_game: pl.DataFrame) -> pl.DataFrame:
    return build_rolling(team_game)


@pytest.fixture(scope="module")
def matrix() -> pl.DataFrame:
    if not FEATURE_MATRIX_PATH.exists():
        pytest.skip("game_features.parquet not built; run python -m nflpred.features.build")
    return pl.read_parquet(FEATURE_MATRIX_PATH)


@pytest.fixture(scope="module")
def sample(rolling: pl.DataFrame) -> pl.DataFrame:
    """Team-games from the matrix era, where a full window is guaranteed."""
    return rolling.filter(pl.col("season") >= MATRIX_START_SEASON).sample(
        SAMPLE_SIZE, seed=SAMPLE_SEED
    )


def _prior_games(team_game: pl.DataFrame, team: str, gameday: object, n: int) -> pl.DataFrame:
    """That team's last ``n`` games before ``gameday``, by an independent path."""
    return (
        team_game.filter((pl.col("team") == team) & (pl.col("gameday") < gameday))
        .sort("gameday", "game_id")
        .tail(n)
    )


# ----------------------------------------------------------- perturbation


@pytest.fixture(scope="module")
def perturbation(team_game: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Rebuild the rolling table with one mid-season box score rewritten.

    The chosen game sits deep enough into a season to have a full window behind
    it and to have games after it, so the test can check both that its own row
    is untouched *and* that later rows do move — an assertion that only holds
    if the perturbation was consequential in the first place.
    """
    target = (
        team_game.filter((pl.col("team") == "KC") & (pl.col("season") == 2015))
        .sort("gameday")
        .row(8, named=True)
    )

    absurd = pl.when(
        (pl.col("game_id") == target["game_id"]) & (pl.col("team") == target["team"])
    )
    perturbed = team_game.with_columns(
        off_epa_per_play=absurd.then(pl.lit(99.0)).otherwise(pl.col("off_epa_per_play")),
        off_success_rate=absurd.then(pl.lit(1.0)).otherwise(pl.col("off_success_rate")),
        off_yards_per_play=absurd.then(pl.lit(50.0)).otherwise(pl.col("off_yards_per_play")),
        points_for=absurd.then(pl.lit(99, dtype=pl.Int32)).otherwise(pl.col("points_for")),
    )

    return build_rolling(team_game), build_rolling(perturbed), target


def test_perturbing_a_game_does_not_change_its_own_features(perturbation):
    """The one test that matters most: a game cannot see itself."""
    before, after, target = perturbation
    key = (pl.col("game_id") == target["game_id"]) & (pl.col("team") == target["team"])

    assert before.filter(key).equals(after.filter(key))


def test_perturbing_a_game_does_change_later_features(perturbation):
    """Guards the test above from passing vacuously."""
    before, after, target = perturbation

    later = (pl.col("team") == target["team"]) & (pl.col("gameday") > target["gameday"])
    changed = before.filter(later).sort("gameday").head(LONGEST_WINDOW)
    rebuilt = after.filter(later).sort("gameday").head(LONGEST_WINDOW)

    assert not changed.equals(rebuilt)
    # ...and specifically in the window that should contain the altered game.
    assert changed["off_epa_r8"].to_list() != rebuilt["off_epa_r8"].to_list()


def test_perturbing_a_game_does_not_change_other_teams(perturbation):
    """A leak across the team partition would show up here."""
    before, after, target = perturbation
    others = pl.col("team") != target["team"]

    assert before.filter(others).equals(after.filter(others))


# ------------------------------------------- full pipeline perturbation


#: A mid-season game with a full window behind it, games after it, and a
#: long-tenured starting quarterback on both sides.
FULL_TARGET_GAME: str = "2015_09_GB_CAR"


@pytest.fixture(scope="module")
def sources() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    if not TEAM_GAME_PATH.exists():
        pytest.skip("team_game.parquet not built; run python -m nflpred.features.pbp_agg")
    return pl.read_parquet(TEAM_GAME_PATH), read_schedules(), read_pbp()


@pytest.fixture(scope="module")
def full_perturbation(sources) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Rebuild the entire matrix with one game rewritten in every source.

    Box score, final score, and quarterback line all move at once, so every
    Phase 3 feature group has something to react to. The game's own row must
    still come back byte-identical.
    """
    team_game, schedules, pbp = sources
    target = (
        schedules.filter(pl.col("game_id") == FULL_TARGET_GAME)
        # `schedules` holds `gameday` as a string; the matrix parses it.
        .with_columns(pl.col("gameday").str.to_date())
        .row(0, named=True)
    )

    in_team_game = pl.col("game_id") == FULL_TARGET_GAME
    absurd_team_game = team_game.with_columns(
        off_epa_per_play=pl.when(in_team_game)
        .then(99.0)
        .otherwise(pl.col("off_epa_per_play")),
        off_success_rate=pl.when(in_team_game).then(1.0).otherwise(pl.col("off_success_rate")),
        off_points=pl.when(in_team_game).then(99).otherwise(pl.col("off_points")),
        off_turnovers=pl.when(in_team_game).then(9).otherwise(pl.col("off_turnovers")),
    )

    # The result, which is what Elo reads and the box score does not touch.
    absurd_schedules = schedules.with_columns(
        home_score=pl.when(in_team_game).then(70).otherwise(pl.col("home_score")),
        away_score=pl.when(in_team_game).then(0).otherwise(pl.col("away_score")),
        result=pl.when(in_team_game).then(70).otherwise(pl.col("result")),
    )

    absurd_pbp = pbp.with_columns(
        qb_epa=pl.when(pl.col("game_id") == FULL_TARGET_GAME)
        .then(99.0)
        .otherwise(pl.col("qb_epa")),
    )

    return (
        build_features(team_game, schedules, pbp),
        build_features(absurd_team_game, absurd_schedules, absurd_pbp),
        target,
    )


def test_full_pipeline_perturbation_leaves_the_games_own_row_untouched(full_perturbation):
    """The strongest check in the suite, now covering Elo and the QB composite."""
    before, after, _ = full_perturbation
    key = pl.col("game_id") == FULL_TARGET_GAME

    features = list(dict.fromkeys([*PHASE3_FEATURES, *wide_features()]))
    assert before.filter(key).select(features).equals(after.filter(key).select(features))


def test_full_pipeline_perturbation_moves_the_next_game(full_perturbation):
    """Guards the test above from passing vacuously, feature group by group."""
    before, after, target = full_perturbation

    involved = pl.col("home_team").is_in([target["home_team"], target["away_team"]]) | (
        pl.col("away_team").is_in([target["home_team"], target["away_team"]])
    )
    later = involved & (pl.col("gameday") > target["gameday"])

    changed = before.filter(later).sort("gameday", "game_id").head(4)
    rebuilt = after.filter(later).sort("gameday", "game_id").head(4)
    assert changed.height == 4

    for column in ("net_epa_r8_diff", "elo_diff", "qb_epa_r8_diff", "turnover_diff_shrunk"):
        assert changed[column].to_list() != rebuilt[column].to_list(), column


# ----------------------------------------------------------- recomputation


def test_rolling_values_recompute_from_prior_games_only(team_game, sample):
    """Independent re-derivation of the same number, for 50 team-games."""
    for row in sample.iter_rows(named=True):
        prior = _prior_games(team_game, row["team"], row["gameday"], LONGEST_WINDOW)
        assert prior.height == LONGEST_WINDOW, f"{row['team']} {row['game_id']}"

        assert row["off_epa_r8"] == pytest.approx(prior["off_epa_per_play"].mean())
        assert row["def_epa_r8"] == pytest.approx(prior["def_epa_per_play"].mean())
        assert row["off_success_r8"] == pytest.approx(prior["off_success_rate"].mean())


def test_short_window_uses_only_the_most_recent_games(team_game, sample):
    """The 4-game window must be the last four, not the first four."""
    for row in sample.iter_rows(named=True):
        prior = _prior_games(team_game, row["team"], row["gameday"], 4)
        assert row["off_epa_r4"] == pytest.approx(prior["off_epa_per_play"].mean())


# ------------------------------------------------------------- chronology


def test_every_contributing_game_kicked_off_earlier(team_game, sample):
    for row in sample.iter_rows(named=True):
        prior = _prior_games(team_game, row["team"], row["gameday"], LONGEST_WINDOW)

        assert prior.height == LONGEST_WINDOW
        assert prior["gameday"].max() < row["gameday"]
        assert row["game_id"] not in prior["game_id"].to_list()


def test_games_in_window_is_full_for_every_matrix_row(rolling):
    """D-6 in one assertion: no 2006+ row is short of history."""
    matrix_era = rolling.filter(pl.col("season") >= MATRIX_START_SEASON)

    for window in ROLLING_WINDOWS:
        assert matrix_era[f"games_in_window_r{window}"].min() == window


# --------------------------------------------------------------- tripwire


def test_no_feature_is_near_perfectly_correlated_with_the_target(matrix):
    """A target-derived column would sit near |1.0|. Real signal sits near 0.2."""
    features = [
        *net_feature_names(),
        *BASELINE_FEATURES,
        *PHASE3_FEATURES,
        *wide_features(),
    ]
    correlations = matrix.select(
        [pl.corr(c, "home_win").abs().alias(c) for c in dict.fromkeys(features)]
    ).row(0, named=True)

    suspicious = {c: r for c, r in correlations.items() if r is not None and r > 0.9}
    assert not suspicious, f"implausibly predictive features: {suspicious}"


def test_market_columns_are_present_but_not_features(matrix):
    """Tier 3 lines ride in the matrix unused, so their value is measurable."""
    assert {"spread_line", "total_line"} <= set(matrix.columns)
    assert not {"spread_line", "total_line"} & set(BASELINE_FEATURES)
    assert not {"spread_line", "total_line"} & set(PHASE3_FEATURES)
    assert not {"spread_line", "total_line"} & set(wide_features())


def test_baseline_features_have_no_nulls(matrix):
    """Nulls here would mean silently dropped games at fit time."""
    for feature in BASELINE_FEATURES:
        assert matrix[feature].null_count() == 0, feature


def test_phase3_features_have_no_nulls_in_any_split(matrix):
    """Checked per split: a feature that is complete on train and full of holes
    on validation would look fine in aggregate and score nonsense."""
    for split in ("train", "calib", "val", "test"):
        section = matrix.filter(pl.col("split") == split)
        assert section.height > 0, split
        for feature in [*PHASE3_FEATURES, *wide_features()]:
            assert section[feature].null_count() == 0, f"{feature} in {split}"


def test_the_baseline_is_still_the_baseline():
    """Phase 3 adds tuples; it does not edit the bar it is measured against."""
    assert BASELINE_FEATURES == (
        "net_epa_r4_diff",
        "net_epa_r8_diff",
        "net_success_r8_diff",
        "net_pass_epa_r8_diff",
        "net_rush_epa_r8_diff",
        "net_yards_per_play_r8_diff",
        "net_explosive_r8_diff",
        "rest_diff",
        "div_game",
        "is_dome",
    )


def test_matrix_row_count_is_unchanged_by_phase_3(matrix):
    """Phase 3 adds columns. It must not add or drop a single game.

    Scoped to completed seasons. The count is a fact about 2006-2025 and would
    otherwise become a calendar bomb the first September the live season starts
    producing results — a test that fails because football happened is a test
    that gets deleted rather than read.
    """
    completed = matrix.filter(pl.col("season") <= LAST_COMPLETED_SEASON)

    assert completed.height == 5431
    assert matrix["game_id"].n_unique() == matrix.height
