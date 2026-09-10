"""The QB composite, with D-8's definition as the thing under test.

Two properties carry the whole module:

* the projected starter for game *N* is the **actual** starter of game *N-1*,
  which is what makes the definition identical at train and serve time;
* a quarterback's rolling form for game *N* excludes game *N*.

Everything else - the backup flag, the unknown flag - is bookkeeping on top of
those two. The backup flag does get its own test, because its whole value is
that it fires on the week *after* a change rather than during it, and that is
exactly the kind of off-by-one a passing type check would never catch.
"""

from __future__ import annotations

import polars as pl
import pytest

from nflpred.config import MATRIX_START_SEASON, TEAM_GAME_PATH
from nflpred.features.qb import NEUTRAL, _starter_by_game, build_qb
from nflpred.ingest import read_pbp

SAMPLE_SEED: int = 20260814
SAMPLE_SIZE: int = 60

pytestmark = pytest.mark.data

#: A game whose starter has a long career behind him and more starts after it.
TARGET_GAME: str = "2015_09_GB_CAR"
TARGET_TEAM: str = "CAR"


@pytest.fixture(scope="module")
def pbp() -> pl.DataFrame:
    return read_pbp()


@pytest.fixture(scope="module")
def team_game() -> pl.DataFrame:
    if not TEAM_GAME_PATH.exists():
        pytest.skip("team_game.parquet not built; run python -m nflpred.features.pbp_agg")
    return pl.read_parquet(TEAM_GAME_PATH)


@pytest.fixture(scope="module")
def starters(pbp: pl.DataFrame) -> pl.DataFrame:
    return _starter_by_game(pbp)


@pytest.fixture(scope="module")
def qb(pbp: pl.DataFrame, team_game: pl.DataFrame) -> pl.DataFrame:
    return build_qb(pbp, team_game)


@pytest.fixture(scope="module")
def matrix_era(qb: pl.DataFrame) -> pl.DataFrame:
    return qb.filter(pl.col("season") >= MATRIX_START_SEASON)


# ---------------------------------------------------------- projected starter


def test_projected_starter_is_the_previous_games_actual_starter(
    qb: pl.DataFrame, starters: pl.DataFrame, team_game: pl.DataFrame
):
    """D-8 in one assertion, checked on 60 real team-games."""
    ordered = (
        team_game.select("game_id", "team", "gameday")
        .with_columns(pl.col("gameday").str.to_date())
        .join(starters.select("game_id", "team", "actual_qb"), on=["game_id", "team"])
        .sort("team", "gameday", "game_id")
    )
    sample = qb.filter(pl.col("season") >= MATRIX_START_SEASON).sample(
        SAMPLE_SIZE, seed=SAMPLE_SEED
    )

    for row in sample.iter_rows(named=True):
        previous = (
            ordered.filter(
                (pl.col("team") == row["team"]) & (pl.col("gameday") < row["gameday"])
            )
            .sort("gameday", "game_id")
            .tail(1)
        )
        assert previous.height == 1, f"{row['team']} {row['game_id']}"
        assert row["projected_qb"] == previous["actual_qb"][0], row["game_id"]


def test_projected_starter_is_never_this_games_own_starter_by_construction(
    qb: pl.DataFrame, starters: pl.DataFrame
):
    """When a team changes quarterback, the projection must be the *old* one.

    If the projection were reading this game's starter, these rows would not
    exist at all - so their existence is the check.
    """
    joined = qb.join(
        starters.select("game_id", "team", "actual_qb"), on=["game_id", "team"], how="inner"
    ).filter(pl.col("projected_qb").is_not_null())

    changes = joined.filter(pl.col("projected_qb") != pl.col("actual_qb"))
    assert changes.height > 500, "no QB changes found - the join is probably wrong"


# ------------------------------------------------------------------- form


def test_qb_form_excludes_the_current_game(
    qb: pl.DataFrame, starters: pl.DataFrame, team_game: pl.DataFrame
):
    """Recompute the 8-start mean from prior starts by an independent path."""
    player_games = (
        team_game.select("game_id", "team", "gameday")
        .with_columns(pl.col("gameday").str.to_date())
        .join(starters, on=["game_id", "team"])
        .sort("gameday", "game_id")
    )

    # Only rows where the projected starter did start, so the expected value is
    # unambiguous: his own last eight starts before this one.
    sample = (
        qb.filter(pl.col("season") >= MATRIX_START_SEASON)
        .join(starters.select("game_id", "team", "actual_qb"), on=["game_id", "team"])
        .filter(
            (pl.col("projected_qb") == pl.col("actual_qb")) & (pl.col("qb_prior_starts") >= 8)
        )
        .sample(SAMPLE_SIZE, seed=SAMPLE_SEED)
    )

    for row in sample.iter_rows(named=True):
        prior = (
            player_games.filter(
                (pl.col("actual_qb") == row["projected_qb"])
                & (pl.col("gameday") < row["gameday"])
            )
            .sort("gameday", "game_id")
            .tail(8)
        )
        assert prior.height == 8, row["game_id"]
        assert row["game_id"] not in prior["game_id"].to_list()
        assert row["qb_epa_r8"] == pytest.approx(prior["qb_game_epa"].mean())


@pytest.fixture(scope="module")
def perturbation(
    pbp: pl.DataFrame, team_game: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Rebuild the composite with one game's quarterback play rewritten."""
    is_target = (pl.col("game_id") == TARGET_GAME) & (pl.col("posteam") == TARGET_TEAM)
    absurd = pbp.with_columns(
        qb_epa=pl.when(is_target).then(99.0).otherwise(pl.col("qb_epa")),
        cpoe=pl.when(is_target).then(99.0).otherwise(pl.col("cpoe")),
    )

    target = (
        team_game.filter(
            (pl.col("game_id") == TARGET_GAME) & (pl.col("team") == TARGET_TEAM)
        )
        .with_columns(pl.col("gameday").str.to_date())
        .row(0, named=True)
    )
    return build_qb(pbp, team_game), build_qb(absurd, team_game), target


def test_perturbing_a_qbs_game_does_not_change_that_games_own_form(perturbation):
    before, after, target = perturbation
    key = (pl.col("game_id") == TARGET_GAME) & (pl.col("team") == TARGET_TEAM)

    assert before.filter(key).equals(after.filter(key))


def test_perturbing_a_qbs_game_does_change_his_next_start(perturbation):
    """Guards the test above from passing vacuously."""
    before, after, target = perturbation
    starter = before.filter(
        (pl.col("game_id") == TARGET_GAME) & (pl.col("team") == TARGET_TEAM)
    )["projected_qb"][0]

    later = (pl.col("projected_qb") == starter) & (pl.col("gameday") > target["gameday"])
    first_before = before.filter(later).sort("gameday", "game_id").head(1)
    first_after = after.filter(later).sort("gameday", "game_id").head(1)

    assert first_before.height == 1
    assert first_before["qb_epa_r8"][0] != first_after["qb_epa_r8"][0]


# ------------------------------------------------------------------ flags


def test_backup_flag_fires_the_week_after_a_change_not_during_it(
    qb: pl.DataFrame, starters: pl.DataFrame
):
    """The 2008 Patriots: Brady hurt in week 1, Cassel from week 2 on.

    Week 1's projection is still Brady - the flag must be *off*, because at
    kickoff nobody knew. Week 2's projection is Cassel and the flag is on.
    """
    season = qb.filter((pl.col("team") == "NE") & (pl.col("season") == 2008)).sort("week")

    assert season["backup_qb_starting"][0] == 0, "flagged the week of the injury"
    assert season["backup_qb_starting"][1] == 1, "not flagged the week after"


@pytest.mark.parametrize(
    ("team", "season", "change_week"),
    [
        ("DEN", 2015, 10),  # Manning -> Osweiler
        ("CAR", 2019, 3),   # Newton -> Allen
        ("SF", 2022, 13),   # Garoppolo -> Purdy
        ("NYJ", 2023, 12),  # Wilson -> Boyle
    ],
)
def test_known_mid_season_changes_flag_one_week_late(qb, team, season, change_week):
    year = qb.filter((pl.col("team") == team) & (pl.col("season") == season)).sort("week")
    weeks = year["week"].to_list()
    flags = year["backup_qb_starting"].to_list()

    assert flags[weeks.index(change_week)] == 0, f"{team} {season} flagged during the change"
    following = weeks[weeks.index(change_week) + 1]
    assert flags[weeks.index(following)] == 1, f"{team} {season} not flagged after"


def test_unknown_form_is_neutral_and_flagged(qb: pl.DataFrame):
    """A projected starter with no prior form is flagged, never averaged over."""
    unknown = qb.filter(pl.col("qb_form_unknown") == 1)
    assert unknown.height > 0

    for column in ("qb_epa_r4", "qb_epa_r8", "qb_cpoe_r4", "qb_cpoe_r8"):
        assert unknown[column].unique().to_list() == [NEUTRAL], column
    assert unknown["qb_prior_starts"].max() == 0


def test_unknown_form_is_structurally_dead_in_the_matrix_era(matrix_era: pl.DataFrame):
    """D-31's second consequence, pinned rather than left to be rediscovered.

    The projected starter *is* whoever started the team's previous game, so once
    the as-of join reads his form **through** that start (rather than up to the
    one before it), he always has at least one start behind him. The flag can
    therefore only fire where a team has no previous game at all - 1999 week 1,
    and Houston's 2002 expansion - which is outside the matrix by D-2.

    It joins the two flags D-24 already found structurally dead. The driver in
    `nflpred.explain.confidence_drivers` is **kept** rather than removed, for D-24's
    stated reason: it fires if that ever stops being true.
    """
    assert matrix_era["qb_form_unknown"].sum() == 0
    assert matrix_era["projected_qb"].null_count() == 0
    assert matrix_era["qb_prior_starts"].min() >= 1


def test_form_columns_have_no_nulls_in_the_matrix_era(matrix_era: pl.DataFrame):
    for column in ("qb_epa_r4", "qb_epa_r8", "qb_cpoe_r4", "qb_cpoe_r8"):
        assert matrix_era[column].null_count() == 0, column
    assert matrix_era["backup_qb_starting"].null_count() == 0
