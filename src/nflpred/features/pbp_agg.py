"""Play-by-play to team-game aggregates.

Produces one row per ``(game_id, team)``: that team's offensive line for the
game, its defensive line (which is definitionally its opponent's offensive
line), and the outcome.

.. warning::
   These rows contain **same-game** box-score statistics. That is correct here
   and is not a leakage violation — this is the raw substrate. Nothing in this
   module may be joined to a game row as a feature. Lagging happens in
   ``nflpred.features.rolling``, which rolls over prior games and
   ``shift(1)``s before joining.

The schedule is the spine rather than the play-by-play, which guarantees every
completed game contributes exactly two rows and gives authoritative home/away
and neutral-site designations.
"""

from __future__ import annotations

import polars as pl

from nflpred.config import EXPLOSIVE_YARDS, POSTSEASON_TYPES

#: Plays counted toward offensive rate statistics, before any win-probability
#: filter.
#:
#: Special teams are excluded because they are a different unit; kneels and
#: spikes because they are clock management and would drag EPA/play down for
#: exactly the teams that are winning. Null EPA means the play was not modelled.
_SCRIMMAGE_PLAY = (
    pl.col("epa").is_not_null()
    & (pl.col("special") == 0)
    & (pl.col("qb_kneel") == 0)
    & (pl.col("qb_spike") == 0)
    & pl.col("posteam").is_not_null()
)


def _scrimmage_play(wp_band: tuple[float, float] | None = None) -> pl.Expr:
    """The play filter, optionally narrowed to a win-probability band (D-9).

    ``wp_band=None`` is the unfiltered Phase 1 behaviour and stays the default:
    per D-3 garbage-time filtering is a modelling assumption to be measured, not
    a cleaning step to be assumed. Passing a band builds the variant.

    ``vegas_wp`` is preferred over ``wp`` because it conditions on the closing
    line, so a 21-point lead against a 14-point favourite reads as less decided
    than the same lead against an underdog. ``wp`` is the fallback.

    **Null win probability keeps the play.** An unjudgeable play is not a
    garbage-time play, and a season where ``vegas_wp`` never populated would
    otherwise have every play dropped and its aggregates silently emptied.
    """
    if wp_band is None:
        return _SCRIMMAGE_PLAY

    low, high = wp_band
    win_probability = pl.coalesce("vegas_wp", "wp")
    competitive = win_probability.is_null() | (
        (win_probability >= low) & (win_probability <= high)
    )
    return _SCRIMMAGE_PLAY & competitive

#: Plays that actually ran from scrimmage (excludes negated-by-penalty plays,
#: which have EPA but no meaningful yardage).
_RAN = (pl.col("pass") == 1) | (pl.col("rush") == 1)

#: Offensive statistics aggregated out of play-by-play.
_PBP_STATS: tuple[str, ...] = (
    "plays",
    "epa_per_play",
    "success_rate",
    "pass_plays",
    "pass_epa_per_play",
    "pass_success_rate",
    "rush_plays",
    "rush_epa_per_play",
    "rush_success_rate",
    "yards_per_play",
    "explosive_rate",
    "third_down_att",
    "third_down_conv",
    "third_down_rate",
    "dropbacks",
    "sacks",
    "sack_rate",
    "interceptions",
    "fumbles_lost",
    "turnovers",
    "qb_epa_per_dropback",
    "cpoe",
    "first_downs",
)

#: Every offensive column, mirrored to ``def_*`` for the opposing side.
#:
#: ``points`` is the odd one out: it comes off the schedule spine rather than
#: out of play-by-play. It is listed here anyway so that ``off_points`` and
#: ``def_points`` ride the existing ``off_``/``def_`` rolling machinery instead
#: of needing a second code path in :mod:`nflpred.features.rolling` — Tier 2's
#: "rolling points scored / allowed" for the cost of an alias.
OFFENSE_STATS: tuple[str, ...] = (*_PBP_STATS, "points")


def _offense_by_game(
    pbp: pl.DataFrame, wp_band: tuple[float, float] | None = None
) -> pl.DataFrame:
    """Aggregate each team's offensive performance within each game.

    Computed once. The defensive side is attached later by joining this frame
    to the opponent, since a team's defensive line *is* its opponent's
    offensive line — there is no second aggregation to do.
    """
    plays = pbp.filter(_scrimmage_play(wp_band))

    return plays.group_by("game_id", "posteam").agg(
        plays=pl.len(),
        epa_per_play=pl.col("epa").mean(),
        success_rate=pl.col("success").mean(),
        # pass/rush splits
        pass_plays=pl.col("pass").sum(),
        pass_epa_per_play=pl.col("epa").filter(pl.col("pass") == 1).mean(),
        pass_success_rate=pl.col("success").filter(pl.col("pass") == 1).mean(),
        rush_plays=pl.col("rush").sum(),
        rush_epa_per_play=pl.col("epa").filter(pl.col("rush") == 1).mean(),
        rush_success_rate=pl.col("success").filter(pl.col("rush") == 1).mean(),
        # yardage — only over plays that actually ran
        yards_per_play=pl.col("yards_gained").filter(_RAN).mean(),
        explosive_rate=(pl.col("yards_gained").filter(_RAN) >= EXPLOSIVE_YARDS).mean(),
        # third down, using nflverse's own flags
        third_down_att=(
            pl.col("third_down_converted").sum() + pl.col("third_down_failed").sum()
        ),
        third_down_conv=pl.col("third_down_converted").sum(),
        # pressure: sack rate is per dropback, not per play
        dropbacks=pl.col("qb_dropback").sum(),
        sacks=pl.col("sack").sum(),
        # turnovers
        interceptions=pl.col("interception").sum(),
        fumbles_lost=pl.col("fumble_lost").sum(),
        # QB quality. cpoe is null before 2006 and on non-pass plays; mean()
        # skips nulls, so pre-2006 rows come out null rather than zero (D-2).
        qb_epa_per_dropback=pl.col("qb_epa").filter(pl.col("qb_dropback") == 1).mean(),
        cpoe=pl.col("cpoe").mean(),
        first_downs=pl.col("first_down").sum(),
    ).with_columns(
        third_down_rate=pl.when(pl.col("third_down_att") > 0)
        .then(pl.col("third_down_conv") / pl.col("third_down_att"))
        .otherwise(None),
        sack_rate=pl.when(pl.col("dropbacks") > 0)
        .then(pl.col("sacks") / pl.col("dropbacks"))
        .otherwise(None),
        turnovers=pl.col("interceptions") + pl.col("fumbles_lost"),
    ).rename({"posteam": "team"})


def _points_from_pbp(pbp: pl.DataFrame) -> pl.DataFrame:
    """Each team's final score, derived independently from play-by-play.

    Exists purely to be reconciled against the schedule's ``home_score`` /
    ``away_score``. A mismatch means the play-by-play join is wrong, and it is
    the strongest single check available at Phase 1.
    """
    as_offense = pbp.select(
        "game_id",
        team=pl.col("posteam"),
        score=pl.col("posteam_score_post"),
    )
    as_defense = pbp.select(
        "game_id",
        team=pl.col("defteam"),
        score=pl.col("defteam_score_post"),
    )
    return (
        pl.concat([as_offense, as_defense])
        .filter(pl.col("team").is_not_null() & pl.col("score").is_not_null())
        .group_by("game_id", "team")
        .agg(pbp_points=pl.col("score").max().cast(pl.Int32))
    )


def _spine(schedules: pl.DataFrame) -> pl.DataFrame:
    """One row per (game, team) for every *completed* game.

    Games with no result — future fixtures, and the cancelled 2022 Bills-Bengals
    game — are dropped here and reported separately by the Phase 1 report.
    """
    played = schedules.filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    )

    shared = ("game_id", "season", "week", "game_type", "gameday")

    home = played.select(
        *shared,
        team=pl.col("home_team"),
        opponent=pl.col("away_team"),
        is_home=pl.lit(1, dtype=pl.Int8),
        points_for=pl.col("home_score").cast(pl.Int32),
        points_against=pl.col("away_score").cast(pl.Int32),
        location=pl.col("location"),
    )
    away = played.select(
        *shared,
        team=pl.col("away_team"),
        opponent=pl.col("home_team"),
        is_home=pl.lit(0, dtype=pl.Int8),
        points_for=pl.col("away_score").cast(pl.Int32),
        points_against=pl.col("home_score").cast(pl.Int32),
        location=pl.col("location"),
    )

    return pl.concat([home, away]).with_columns(
        # `is_home` stays nominal even at neutral sites — the Super Bowl still
        # has a designated home team for line purposes. Whether the home-field
        # feature should be zeroed there is a Phase 3 call, so both facts are
        # recorded and neither is baked in.
        is_neutral_site=(pl.col("location") == "Neutral").cast(pl.Int8),
        is_postseason=pl.col("game_type").is_in(POSTSEASON_TYPES).cast(pl.Int8),
        margin=pl.col("points_for") - pl.col("points_against"),
    ).with_columns(
        # Ties are kept, not dropped (D-4). `won` is a float so a tie is 0.5;
        # `result_class` keeps the three-way distinction legible.
        won=pl.when(pl.col("margin") > 0)
        .then(1.0)
        .when(pl.col("margin") < 0)
        .then(0.0)
        .otherwise(0.5),
        result_class=pl.when(pl.col("margin") > 0)
        .then(pl.lit("win"))
        .when(pl.col("margin") < 0)
        .then(pl.lit("loss"))
        .otherwise(pl.lit("tie")),
    ).drop("location")


def build_team_game(
    pbp: pl.DataFrame,
    schedules: pl.DataFrame,
    wp_band: tuple[float, float] | None = None,
) -> pl.DataFrame:
    """Assemble the tidy team-game table.

    ``wp_band`` narrows the play filter to a win-probability window, producing
    the garbage-time variant (D-9). The default is unfiltered; both tables are
    built and neither is privileged until validation picks one.
    """
    offense = _offense_by_game(pbp, wp_band)
    spine = _spine(schedules)

    off_cols = {stat: f"off_{stat}" for stat in _PBP_STATS}
    def_cols = {stat: f"def_{stat}" for stat in _PBP_STATS}

    team_game = (
        spine
        # this team's own offence
        .join(
            offense.rename(off_cols),
            on=["game_id", "team"],
            how="left",
        )
        # the opponent's offence, which is this team's defence
        .join(
            offense.rename(def_cols).rename({"team": "opponent"}),
            on=["game_id", "opponent"],
            how="left",
        )
        .join(
            _points_from_pbp(pbp),
            on=["game_id", "team"],
            how="left",
        )
        # Aliases, not new statistics: `points_for`/`points_against` already
        # exist above and are authoritative. These names are what let the
        # rolling machinery pick them up alongside the pbp-derived columns.
        .with_columns(
            off_points=pl.col("points_for"),
            def_points=pl.col("points_against"),
        )
    )

    ordered = [
        "game_id",
        "season",
        "week",
        "game_type",
        "gameday",
        "team",
        "opponent",
        "is_home",
        "is_neutral_site",
        "is_postseason",
        "points_for",
        "points_against",
        "margin",
        "won",
        "result_class",
        "pbp_points",
        *[f"off_{s}" for s in OFFENSE_STATS],
        *[f"def_{s}" for s in OFFENSE_STATS],
    ]
    return team_game.select(ordered).sort("season", "week", "game_id", "team")


def main() -> None:
    """Build the team-game table from the cache and write it to processed/."""
    import argparse

    # Imported here so the aggregation stays testable against synthetic frames
    # without pulling in the network-facing module.
    from nflpred.config import TEAM_GAME_GT_PATH, TEAM_GAME_PATH, WP_BAND
    from nflpred.ingest import read_pbp, read_schedules

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--garbage-time",
        action="store_true",
        help=f"drop plays outside win probability {WP_BAND}, writing the _gt table (D-9)",
    )
    args = parser.parse_args()

    wp_band = WP_BAND if args.garbage_time else None
    path = TEAM_GAME_GT_PATH if args.garbage_time else TEAM_GAME_PATH

    team_game = build_team_game(read_pbp(), read_schedules(), wp_band)
    path.parent.mkdir(parents=True, exist_ok=True)
    team_game.write_parquet(path, compression="zstd")

    print(
        f"team_game: {team_game.height:,} rows x {team_game.width} cols "
        f"-> {path.relative_to(path.parents[2])} "
        f"({path.stat().st_size / 1e6:.1f} MB)"
    )
    if wp_band:
        print(f"  win-probability band {wp_band}; nulls kept")
        print(f"  mean plays/team-game: {team_game['off_plays'].mean():.1f}")


if __name__ == "__main__":
    main()
