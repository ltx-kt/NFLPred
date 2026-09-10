"""Elo, with the ordering rule as the thing under test.

:mod:`nflpred.features.elo` records a pre-game rating and only then applies the
update. Every other property of the system — that ratings regress, that the
expectation is symmetric, that HOU starts low — is arithmetic that would be
merely wrong if broken. That one ordering rule is the difference between a
feature and a leak, so it gets the perturbation treatment: rewrite a game's
result, rebuild, and demand that the game's *own* rating did not move while the
next one's did.

Runs against real schedules rather than synthetic ones. A leak at the seam
between two seasons, or on the one expansion franchise in range, would never
show up in a toy frame.
"""

from __future__ import annotations

import polars as pl
import pytest

from nflpred.config import (
    ELO_EXPANSION_INIT,
    ELO_INIT,
    ELO_SEASON_REGRESSION,
    MATRIX_START_SEASON,
)
from nflpred.features.elo import build_elo, expected_score, season_end_ratings
from nflpred.ingest import SCHEDULES_PATH, read_schedules

#: A mid-season game with plenty of history behind it and a season after it.
TARGET_GAME: str = "2015_09_GB_CAR"


@pytest.fixture(scope="module")
def schedules() -> pl.DataFrame:
    if not SCHEDULES_PATH.exists():
        pytest.skip("schedules.parquet not cached; run python -m nflpred.ingest")
    return read_schedules()


@pytest.fixture(scope="module")
def elo(schedules: pl.DataFrame) -> pl.DataFrame:
    return build_elo(schedules)


# ----------------------------------------------------------- perturbation


@pytest.fixture(scope="module")
def perturbation(schedules: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Rebuild the ratings with one game's result rewritten to a blowout."""
    target = (
        schedules.filter(pl.col("game_id") == TARGET_GAME)
        # `schedules` carries `gameday` as a string; the Elo frame parses it.
        # Comparing the two later needs them to be the same kind of thing.
        .with_columns(pl.col("gameday").str.to_date())
        .row(0, named=True)
    )
    assert target["home_score"] is not None, "target game must have a result"

    is_target = pl.col("game_id") == TARGET_GAME
    flipped = schedules.with_columns(
        home_score=pl.when(is_target).then(70).otherwise(pl.col("home_score")),
        away_score=pl.when(is_target).then(0).otherwise(pl.col("away_score")),
        result=pl.when(is_target).then(70).otherwise(pl.col("result")),
    )

    return build_elo(schedules), build_elo(flipped), target


def test_perturbing_a_result_does_not_change_that_games_own_elo(perturbation):
    """The rating is recorded before the update. This is that, in one assertion."""
    before, after, _ = perturbation
    key = pl.col("game_id") == TARGET_GAME

    assert before.filter(key).equals(after.filter(key))


def test_perturbing_a_result_does_change_the_next_games_elo(perturbation):
    """Guards the test above from passing vacuously."""
    before, after, target = perturbation

    for team in (target["home_team"], target["away_team"]):
        later = (pl.col("team") == team) & (pl.col("gameday") > target["gameday"])
        first_before = before.filter(later).sort("gameday", "game_id").head(1)
        first_after = after.filter(later).sort("gameday", "game_id").head(1)

        assert first_before.height == 1
        assert first_before["elo_pre"][0] != first_after["elo_pre"][0], team


def test_perturbing_a_result_does_not_change_uninvolved_teams_before_they_meet(
    perturbation,
):
    """Elo is transitive, so a shock propagates — but not backwards in time."""
    before, after, target = perturbation
    involved = {target["home_team"], target["away_team"]}

    earlier = pl.col("gameday") <= target["gameday"]
    others = ~pl.col("team").is_in(list(involved))

    assert before.filter(earlier & others).equals(after.filter(earlier & others))


# --------------------------------------------------------------- structure


def test_two_rows_per_game_with_mirrored_ratings(elo):
    """Each game emits both perspectives, and they agree with each other."""
    paired = elo.join(
        elo.select("game_id", opponent="team", opp_elo="elo_pre", opp_prob="elo_prob"),
        on=["game_id", "opponent"],
    )

    assert (paired["elo_opp_pre"] - paired["opp_elo"]).abs().max() == pytest.approx(0.0)
    assert (paired["elo_prob"] + paired["opp_prob"] - 1.0).abs().max() == pytest.approx(0.0)
    assert (paired["elo_diff"] + (paired["opp_elo"] - paired["elo_pre"])).abs().max() == (
        pytest.approx(0.0)
    )


def test_expected_score_is_a_symmetric_probability():
    assert expected_score(1500, 1500) == pytest.approx(0.5)
    assert expected_score(1500, 1500, advantage=55) > 0.5
    assert expected_score(1900, 1500) + expected_score(1500, 1900) == pytest.approx(1.0)
    # The 400-point convention: a 400-point edge is 10:1.
    assert expected_score(1900, 1500) == pytest.approx(10 / 11)


def test_home_advantage_applies_only_at_non_neutral_sites(elo):
    """A neutral-site game gives the nominal home team no Elo bonus."""
    neutral = elo.filter((pl.col("is_neutral_site") == 1) & (pl.col("is_home") == 1))
    assert neutral.height > 0

    implied = neutral.select(
        expected=pl.col("elo_prob"),
        bare=1.0 / (1.0 + 10.0 ** ((pl.col("elo_opp_pre") - pl.col("elo_pre")) / 400.0)),
    )
    assert (implied["expected"] - implied["bare"]).abs().max() == pytest.approx(0.0)


def test_expansion_team_starts_below_the_mean(elo):
    """HOU 2002 is the only expansion franchise inside the ingest range."""
    first = elo.filter(pl.col("team") == "HOU").sort("gameday", "game_id").head(1)

    assert first["season"][0] == 2002
    assert first["elo_pre"][0] == pytest.approx(ELO_EXPANSION_INIT)


def test_ratings_regress_toward_the_mean_between_seasons(schedules, elo):
    """A team's first rating of a season is its last one, pulled a third in."""
    finals = season_end_ratings(schedules)
    openers = (
        elo.sort("gameday", "game_id")
        .group_by("team", "season")
        .agg(pl.col("elo_pre").first())
    )

    joined = (
        openers.join(
            finals.with_columns(season=pl.col("season") + 1),
            on=["team", "season"],
            how="inner",
        )
        .filter(pl.col("season") > 2002)  # after every franchise has a history
    )
    assert joined.height > 500

    expected = ELO_INIT + (1.0 - ELO_SEASON_REGRESSION) * (joined["elo"] - ELO_INIT)
    assert (joined["elo_pre"] - expected).abs().max() == pytest.approx(0.0, abs=1e-9)


def test_every_matrix_era_game_has_a_rating(elo):
    """Nothing is null and nothing is still sitting at its initial value."""
    matrix_era = elo.filter(pl.col("season") >= MATRIX_START_SEASON)

    assert matrix_era["elo_pre"].null_count() == 0
    assert (matrix_era["elo_pre"] == ELO_INIT).sum() == 0
    assert matrix_era["elo_pre"].min() > 1000
    assert matrix_era["elo_pre"].max() < 2000


# ----------------------------------------------------------- face validity


@pytest.mark.parametrize(
    ("season", "team", "position"),
    [
        (2007, "NE", "top"),      # 16-0
        (2008, "DET", "bottom"),  # 0-16
        (2013, "SEA", "top"),     # Super Bowl XLVIII
        (2017, "CLE", "bottom"),  # 0-16
    ],
)
def test_known_seasons_land_where_history_says(schedules, season, team, position):
    """A rating system that disagrees with history is broken whatever it scores."""
    year = season_end_ratings(schedules).filter(pl.col("season") == season)
    ranked = year.sort("elo", descending=(position == "top"))["team"].to_list()

    assert team in ranked[:3], f"{team} {season} ranked {ranked.index(team) + 1} from {position}"
