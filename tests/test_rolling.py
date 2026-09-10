"""Unit tests for the lagging machinery.

The synthetic series here use values chosen so that a hand-computed mean is
unambiguous — if a window is off by one game, or the shift is missing, the
arithmetic stops matching immediately.
"""

from __future__ import annotations

import polars as pl
import pytest

from nflpred.config import ROLLING_WINDOWS
from nflpred.features.rolling import build_rolling, rolled_columns, std_columns

WINDOWS = (3,)
STATS = ("epa_per_play",)

#: The synthetic series carries one statistic, so the season-to-date window is
#: restricted to it too — the real :data:`~nflpred.config.STD_WINDOW_STATS` names
#: columns these toy frames do not have.
STD_STATS = STATS


def _series(values: list[float], team: str = "AAA", start_season: int = 2023) -> pl.DataFrame:
    """A team's game log, one game per week, with a known statistic per game.

    Weeks run past 17 into a second season so a boundary crossing can be tested
    without inventing a second team.
    """
    rows = []
    for i, value in enumerate(values):
        season = start_season + i // 17
        week = i % 17 + 1
        rows.append(
            {
                "game_id": f"{season}_{week:02d}_{team}_OPP",
                "season": season,
                "week": week,
                "game_type": "REG",
                # Weeks are 7 days apart; a new season starts the following year.
                "gameday": f"{season}-09-{week + 1:02d}",
                "team": team,
                "opponent": "OPP",
                "is_home": 1,
                "is_neutral_site": 0,
                "is_postseason": 0,
                "won": 1.0,
                "off_epa_per_play": value,
                "def_epa_per_play": -value,
            }
        )
    return pl.DataFrame(rows)


def _rolled(frame: pl.DataFrame) -> pl.DataFrame:
    return build_rolling(
        frame, windows=WINDOWS, stats=STATS, std_stats=STD_STATS
    ).sort("gameday")


def test_rolling_value_is_the_mean_of_the_prior_window():
    """Game N carries the mean of N-3..N-1 and never touches N itself."""
    out = _rolled(_series([1.0, 2.0, 3.0, 4.0, 5.0]))

    assert out["off_epa_r3"].to_list() == [
        None,                 # no history
        1.0,                  # mean(1)
        pytest.approx(1.5),   # mean(1, 2)
        2.0,                  # mean(1, 2, 3)
        3.0,                  # mean(2, 3, 4) — the 1.0 has fallen out
    ]


def test_the_current_game_never_contributes():
    """Changing only the last game's value must not change its own feature."""
    baseline = _rolled(_series([1.0, 2.0, 3.0, 4.0]))
    perturbed = _rolled(_series([1.0, 2.0, 3.0, 999.0]))

    assert baseline["off_epa_r3"].to_list() == perturbed["off_epa_r3"].to_list()


def test_first_game_of_a_series_is_null_not_zero():
    """A team with no history has no form. Zero would be a fabricated average."""
    out = _rolled(_series([0.4, 0.5]))

    assert out["off_epa_r3"][0] is None
    assert out["def_epa_r3"][0] is None


def test_defensive_side_rolls_independently():
    out = _rolled(_series([1.0, 2.0, 3.0, 4.0]))

    assert out["def_epa_r3"].to_list() == [None, -1.0, pytest.approx(-1.5), -2.0]


def test_window_carries_across_a_season_boundary():
    """D-6: a new season's week 1 uses the tail of the previous season."""
    # 17 games in 2023, then the 2024 opener.
    values = [1.0] * 15 + [10.0, 20.0, 30.0]
    out = _rolled(_series(values))

    opener = out.filter((pl.col("season") == 2024) & (pl.col("week") == 1))
    assert opener.height == 1
    # mean of the last three 2023 games (1.0, 10.0, 20.0) — not null, not reset.
    assert opener["off_epa_r3"][0] == pytest.approx(31.0 / 3)
    assert opener["games_in_window_r3"][0] == 3


def test_games_in_window_counts_prior_games_only():
    out = _rolled(_series([1.0, 2.0, 3.0, 4.0, 5.0]))

    assert out["games_in_window_r3"].to_list() == [0, 1, 2, 3, 3]


def test_teams_do_not_bleed_into_each_other():
    """One team's history must never appear in another team's window."""
    frame = pl.concat([_series([1.0, 1.0, 1.0], team="AAA"),
                       _series([9.0, 9.0, 9.0], team="BBB")])
    out = build_rolling(frame, windows=WINDOWS, stats=STATS, std_stats=STD_STATS)

    bbb = out.filter(pl.col("team") == "BBB").sort("gameday")
    assert bbb["off_epa_r3"].to_list() == [None, 9.0, 9.0]


def test_ordering_is_by_date_not_week():
    """Postseason week numbers restart low; date ordering must still hold.

    A week-1 playoff-style row dated after the regular season has to sort last,
    otherwise its result would leak backwards into earlier games.
    """
    frame = _series([1.0, 2.0])
    late = frame[1].with_columns(
        game_id=pl.lit("2023_01_AAA_LATE"),
        week=pl.lit(1, dtype=pl.Int64),
        gameday=pl.lit("2024-01-14"),
        off_epa_per_play=pl.lit(5.0),
        def_epa_per_play=pl.lit(-5.0),
    )
    out = build_rolling(
        pl.concat([frame, late]), windows=WINDOWS, stats=STATS, std_stats=STD_STATS
    )

    playoff = out.filter(pl.col("game_id") == "2023_01_AAA_LATE")
    # Sees both regular-season games; contributes to neither.
    assert playoff["off_epa_r3"][0] == pytest.approx(1.5)
    assert playoff["games_in_window_r3"][0] == 2


def test_emits_a_column_per_stat_side_and_window():
    frame = _series([1.0, 2.0, 3.0])
    out = build_rolling(
        frame, windows=ROLLING_WINDOWS, stats=STATS, std_stats=STD_STATS
    )

    expected = set(rolled_columns(ROLLING_WINDOWS, STATS))
    assert expected == {"off_epa_r4", "off_epa_r8", "def_epa_r4", "def_epa_r8"}
    assert expected <= set(out.columns)
    assert all(f"games_in_window_r{w}" in out.columns for w in ROLLING_WINDOWS)
    assert set(std_columns(STD_STATS)) == {"off_epa_std", "def_epa_std"}
    assert set(std_columns(STD_STATS)) <= set(out.columns)


# ------------------------------------------------------- season-to-date


def test_season_to_date_is_the_expanding_mean_of_prior_games_this_season():
    """Game N carries the mean of this season's games 1..N-1, never N."""
    out = _rolled(_series([1.0, 2.0, 3.0, 4.0]))

    # Week 1 falls back to the cross-season window, which is null with no
    # history at all; weeks 2+ expand rather than sliding.
    assert out["off_epa_std"].to_list() == [
        None,                 # week 1: nothing this season and nothing before
        1.0,                  # mean(1)
        pytest.approx(1.5),   # mean(1, 2)
        2.0,                  # mean(1, 2, 3) — the 1.0 has *not* fallen out
    ]


def test_season_to_date_resets_each_season():
    """Unlike the fixed windows, this one does not cross D-6's boundary."""
    values = [1.0] * 17 + [50.0, 60.0]
    out = _rolled(_series(values))

    second = out.filter(pl.col("season") == 2024).sort("week")
    # Week 1 has no season-to-date history, so it carries the 3-game window.
    assert second["std_is_fallback"].to_list() == [1, 0]
    assert second["off_epa_std"][0] == pytest.approx(1.0)  # from the r3 fallback
    assert second["off_epa_std"][1] == pytest.approx(50.0)  # only this season


def test_season_to_date_never_includes_the_current_game():
    baseline = _rolled(_series([1.0, 2.0, 3.0, 4.0]))
    perturbed = _rolled(_series([1.0, 2.0, 3.0, 999.0]))

    assert baseline["off_epa_std"].to_list() == perturbed["off_epa_std"].to_list()


def test_games_in_season_counts_prior_games_within_the_season():
    out = _rolled(_series([1.0] * 19))

    counts = out.sort("season", "week")["games_in_season"].to_list()
    assert counts[:3] == [0, 1, 2]
    assert counts[17:] == [0, 1]  # the 2024 opener restarts the count
