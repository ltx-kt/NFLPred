"""Structural invariants for the team-game table.

Two kinds of test here. The synthetic ones pin down the rate denominators
against hand-built play-by-play, so a change in filtering fails loudly. The
rest assert invariants over the real built table - cheap, and they catch join
errors that unit tests on toy data never would.
"""

from __future__ import annotations

import polars as pl
import pytest

from nflpred.config import MATRIX_START_SEASON, TEAM_ALIASES, TEAM_GAME_PATH
from nflpred.features.pbp_agg import build_team_game

#: Games present in `load_schedules` with a final score but no play-by-play at
#: all. Upstream gaps, all pre-2006 and therefore outside the matrix (D-2).
GAMES_WITHOUT_PBP: frozenset[str] = frozenset({
    "1999_01_BAL_STL",
    "2000_03_SD_KC",
    "2000_06_BUF_MIA",
})

#: Games whose play-by-play score columns disagree with the schedule's final
#: score. The schedule is authoritative and supplies the label, so these do not
#: affect the target - `pbp_points` is only a cross-check. Pinned so that a
#: *new* mismatch, which would mean a broken join, still fails the suite.
GAMES_WITH_BAD_PBP_SCORES: frozenset[str] = frozenset({
    "2001_01_PIT_JAX", "2001_02_TEN_JAX", "2001_03_CLE_JAX", "2001_06_BUF_JAX",
    "2001_09_CIN_JAX", "2001_11_BAL_JAX", "2001_12_GB_JAX", "2001_16_KC_JAX",
    "2002_01_IND_JAX", "2002_04_NYJ_JAX", "2002_05_PHI_JAX", "2002_08_HOU_JAX",
    "2002_10_WAS_JAX", "2002_13_PIT_JAX", "2002_14_CLE_JAX", "2002_16_TEN_JAX",
    "2011_13_DET_NO",
})

# --------------------------------------------------------------- synthetic


def _play(**overrides) -> dict:
    """A single play-by-play row with neutral defaults."""
    row = {
        "game_id": "2024_01_AAA_BBB",
        "play_id": 1.0,
        "season": 2024,
        "season_type": "REG",
        "week": 1,
        "posteam": "BBB",
        "defteam": "AAA",
        "posteam_type": "home",
        "home_team": "BBB",
        "away_team": "AAA",
        "play": 1.0,
        "special": 0.0,
        "pass": 0.0,
        "rush": 1.0,
        "qb_dropback": 0.0,
        "qb_kneel": 0.0,
        "qb_spike": 0.0,
        "down": 1.0,
        "ydstogo": 10.0,
        "epa": 0.1,
        "success": 1.0,
        "yards_gained": 5.0,
        "first_down": 0.0,
        "qb_epa": 0.1,
        "cpoe": None,
        "passer_player_id": None,
        "third_down_converted": 0.0,
        "third_down_failed": 0.0,
        "sack": 0.0,
        "interception": 0.0,
        "fumble_lost": 0.0,
        "posteam_score_post": 0.0,
        "defteam_score_post": 0.0,
        "touchdown": 0.0,
    }
    row.update(overrides)
    return row


def _schedule(**overrides) -> pl.DataFrame:
    row = {
        "game_id": "2024_01_AAA_BBB",
        "season": 2024,
        "game_type": "REG",
        "week": 1,
        "gameday": "2024-09-08",
        "home_team": "BBB",
        "away_team": "AAA",
        "home_score": 20,
        "away_score": 17,
        "location": "Home",
    }
    row.update(overrides)
    return pl.DataFrame([row])


def test_rate_denominators_exclude_non_plays():
    """Kneels, spikes, special teams and null-EPA plays must not count."""
    pbp = pl.DataFrame([
        _play(play_id=1.0),                      # counts
        _play(play_id=2.0),                      # counts
        _play(play_id=3.0, qb_kneel=1.0),        # excluded
        _play(play_id=4.0, qb_spike=1.0),        # excluded
        _play(play_id=5.0, special=1.0),         # excluded
        _play(play_id=6.0, epa=None),            # excluded
    ])

    out = build_team_game(pbp, _schedule())
    offense = out.filter(pl.col("team") == "BBB")

    assert offense["off_plays"][0] == 2


def test_sack_rate_is_per_dropback_not_per_play():
    pbp = pl.DataFrame([
        _play(play_id=1.0, **{"pass": 1.0}, rush=0.0, qb_dropback=1.0, sack=1.0),
        _play(play_id=2.0, **{"pass": 1.0}, rush=0.0, qb_dropback=1.0),
        _play(play_id=3.0, rush=1.0),  # not a dropback
        _play(play_id=4.0, rush=1.0),
    ])

    offense = build_team_game(pbp, _schedule()).filter(pl.col("team") == "BBB")

    assert offense["off_dropbacks"][0] == 2
    assert offense["off_sack_rate"][0] == pytest.approx(0.5)  # 1 of 2 dropbacks


def test_explosive_rate_counts_only_plays_that_ran():
    pbp = pl.DataFrame([
        _play(play_id=1.0, yards_gained=25.0),  # explosive
        _play(play_id=2.0, yards_gained=3.0),
        _play(play_id=3.0, yards_gained=0.0, **{"pass": 0.0}, rush=0.0),  # negated
    ])

    offense = build_team_game(pbp, _schedule()).filter(pl.col("team") == "BBB")

    assert offense["off_explosive_rate"][0] == pytest.approx(0.5)


def test_defense_mirrors_opponent_offense():
    pbp = pl.DataFrame([
        _play(play_id=1.0, posteam="BBB", defteam="AAA", epa=0.5),
        _play(play_id=2.0, posteam="AAA", defteam="BBB", epa=-0.3),
    ])

    out = build_team_game(pbp, _schedule())
    home = out.filter(pl.col("team") == "BBB")
    away = out.filter(pl.col("team") == "AAA")

    assert home["off_epa_per_play"][0] == pytest.approx(away["def_epa_per_play"][0])
    assert away["off_epa_per_play"][0] == pytest.approx(home["def_epa_per_play"][0])


def test_tie_is_kept_as_half():
    pbp = pl.DataFrame([_play()])
    out = build_team_game(pbp, _schedule(home_score=20, away_score=20))

    assert out["won"].to_list() == [0.5, 0.5]
    assert set(out["result_class"].to_list()) == {"tie"}


def test_neutral_site_keeps_nominal_home_flag():
    pbp = pl.DataFrame([_play()])
    out = build_team_game(pbp, _schedule(location="Neutral"))

    assert out["is_neutral_site"].to_list() == [1, 1]
    assert sorted(out["is_home"].to_list()) == [0, 1]


# -------------------------------------------------------------- real table


@pytest.fixture(scope="module")
def team_game() -> pl.DataFrame:
    if not TEAM_GAME_PATH.exists():
        pytest.skip("team_game.parquet not built; run python -m nflpred.features.pbp_agg")
    return pl.read_parquet(TEAM_GAME_PATH)


def test_every_game_has_exactly_two_rows(team_game):
    counts = team_game.group_by("game_id").len()
    assert counts.filter(pl.col("len") != 2).height == 0


def test_no_duplicate_game_team(team_game):
    assert team_game.select("game_id", "team").is_duplicated().sum() == 0


def test_is_home_sums_to_one_per_game(team_game):
    sums = team_game.group_by("game_id").agg(pl.col("is_home").sum())
    assert sums.filter(pl.col("is_home") != 1).height == 0


def test_points_from_pbp_match_schedule_scores(team_game):
    """The strongest Phase 1 check: two independent paths to the same score."""
    mismatched = team_game.filter(
        pl.col("pbp_points").is_not_null()
        & (pl.col("pbp_points") != pl.col("points_for"))
    )
    unexpected = set(mismatched["game_id"].unique()) - GAMES_WITH_BAD_PBP_SCORES
    assert not unexpected, f"new score mismatches: {sorted(unexpected)}"

    # 99.5%+ of rows must reconcile, or something structural has broken.
    assert mismatched.height / team_game.height < 0.005


def test_margins_are_antisymmetric(team_game):
    paired = team_game.join(
        team_game.select("game_id", opponent="team", opp_margin="margin"),
        on=["game_id", "opponent"],
    )
    assert paired.filter(pl.col("margin") != -pl.col("opp_margin")).height == 0


def test_won_encoding_matches_result_class(team_game):
    assert set(team_game["won"].unique().to_list()) <= {0.0, 0.5, 1.0}
    bad = team_game.filter(
        ((pl.col("result_class") == "win") & (pl.col("won") != 1.0))
        | ((pl.col("result_class") == "loss") & (pl.col("won") != 0.0))
        | ((pl.col("result_class") == "tie") & (pl.col("won") != 0.5))
    )
    assert bad.height == 0


def test_cpoe_absent_before_2006_and_present_after(team_game):
    """Guards decision D-2: the reason the matrix starts at 2006."""
    early = team_game.filter(pl.col("season") < MATRIX_START_SEASON)
    late = team_game.filter(pl.col("season") >= MATRIX_START_SEASON)

    assert early["off_cpoe"].is_null().all()
    # Five 2006 games at Arrowhead have no charting data upstream - 0.1% of rows.
    assert late["off_cpoe"].null_count() / late.height < 0.005


def test_no_rows_without_play_by_play(team_game):
    """A completed game with no aggregated plays means a broken join.

    Regression guard for the relocated-franchise bug: `schedules` used OAK/SD/
    STL while `pbp` used LV/LAC/LA, which silently dropped ~880 games until
    `normalize_teams` reconciled the two vocabularies.
    """
    orphaned = set(
        team_game.filter(pl.col("off_plays").is_null())["game_id"].unique()
    )
    unexpected = orphaned - GAMES_WITHOUT_PBP
    assert orphaned == GAMES_WITHOUT_PBP, f"unexpected orphans: {sorted(unexpected)}"


def test_matrix_seasons_have_complete_play_by_play(team_game):
    """Every game from 2006 on must have both sides aggregated."""
    matrix = team_game.filter(pl.col("season") >= MATRIX_START_SEASON)
    assert matrix.filter(pl.col("off_plays").is_null()).height == 0
    assert matrix.filter(pl.col("def_plays").is_null()).height == 0


def test_no_legacy_team_codes_survive(team_game):
    """Relocated franchises must appear under one code across all seasons."""
    codes = set(team_game["team"].unique()) | set(team_game["opponent"].unique())
    assert not (codes & set(TEAM_ALIASES)), f"un-normalised: {codes & set(TEAM_ALIASES)}"
    assert len(codes) == 32
