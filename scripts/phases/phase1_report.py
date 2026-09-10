"""Phase 1 checkpoint report.

Prints what the spec asks for at the end of the data phase - shape, date range,
null counts - plus the reconciliation checks that decide whether the team-game
table can be trusted as the substrate for everything downstream.

    python scripts/phase1_report.py
"""

from __future__ import annotations

import polars as pl

from nflpred.config import (
    DATA_RAW,
    MATRIX_START_SEASON,
    TEAM_GAME_PATH,
    TEST_SEASONS,
    TRAIN_SEASONS,
    VAL_SEASONS,
    season_split,
)
from nflpred.features.pbp_agg import OFFENSE_STATS
from nflpred.ingest import read_schedules, read_team_stats

pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_cols(12)
pl.Config.set_tbl_width_chars(120)


def header(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    team_game = pl.read_parquet(TEAM_GAME_PATH)
    schedules = read_schedules()

    # ------------------------------------------------------------ cache
    header("1. CACHE")
    files = sorted(DATA_RAW.rglob("*.parquet"))
    total_mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"  raw files       {len(files)}")
    print(f"  raw size        {total_mb:.1f} MB")
    print(f"  team_game       {TEAM_GAME_PATH.stat().st_size / 1e6:.1f} MB")

    # ------------------------------------------------------------ shape
    header("2. SHAPE AND COVERAGE")
    print(f"  rows            {team_game.height:,}")
    print(f"  columns         {team_game.width}")
    print(f"  games           {team_game['game_id'].n_unique():,}")
    print(f"  teams           {team_game['team'].n_unique()}")
    print(f"  seasons         {team_game['season'].min()}-{team_game['season'].max()}")
    print(f"  date range      {team_game['gameday'].min()} to {team_game['gameday'].max()}")
    print(f"  postseason      {team_game['is_postseason'].sum():,} rows")
    print(f"  neutral site    {team_game['is_neutral_site'].sum():,} rows")
    print(f"  ties            {team_game.filter(pl.col('result_class') == 'tie').height} rows")

    header("3. SPLIT ALLOCATION (games, not rows)")
    alloc = (
        team_game.with_columns(
            split=pl.col("season").map_elements(season_split, return_dtype=pl.String)
        )
        .group_by("split")
        .agg(
            games=(pl.col("game_id").n_unique()),
            seasons=pl.col("season").n_unique(),
            first=pl.col("season").min(),
            last=pl.col("season").max(),
        )
        .sort("first")
    )
    print(alloc)
    print(f"\n  train {TRAIN_SEASONS}  val {VAL_SEASONS}  test {TEST_SEASONS}")
    print(f"  'pre' = ingested, excluded from the matrix (D-2, cpoe starts {MATRIX_START_SEASON})")

    # ------------------------------------------------------ excluded games
    header("4. GAMES IN SCHEDULES BUT NOT IN THE TABLE")
    no_result = schedules.filter(
        pl.col("home_score").is_null() | pl.col("away_score").is_null()
    )
    print(f"  {no_result.height} scheduled games have no final score")
    print(no_result.group_by("season").len().sort("season"))
    cancelled = no_result.filter(pl.col("season") <= 2025)
    if cancelled.height:
        print("\n  in a completed season (cancelled/unplayed):")
        print(cancelled.select("game_id", "season", "week", "away_team", "home_team"))

    # --------------------------------------------------- structural checks
    header("5. STRUCTURAL CHECKS")
    per_game = team_game.group_by("game_id").len()
    home_sum = team_game.group_by("game_id").agg(pl.col("is_home").sum())
    checks = [
        ("every game has exactly 2 rows", per_game.filter(pl.col("len") != 2).height == 0),
        ("is_home sums to 1 per game", home_sum.filter(pl.col("is_home") != 1).height == 0),
        (
            "no duplicate (game_id, team)",
            team_game.select("game_id", "team").is_duplicated().sum() == 0,
        ),
        (
            "32 team codes, no legacy aliases",
            team_game["team"].n_unique() == 32,
        ),
        (
            f"complete pbp from {MATRIX_START_SEASON}",
            team_game.filter(
                (pl.col("season") >= MATRIX_START_SEASON) & pl.col("off_plays").is_null()
            ).height
            == 0,
        ),
    ]
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {label}")

    # ------------------------------------------------ score reconciliation
    header("6. SCORE RECONCILIATION (pbp-derived vs schedule, independent paths)")
    scored = team_game.filter(pl.col("pbp_points").is_not_null())
    mismatch = scored.filter(pl.col("pbp_points") != pl.col("points_for"))
    print(f"  rows compared   {scored.height:,}")
    print(f"  agreeing        {scored.height - mismatch.height:,} "
          f"({100 * (1 - mismatch.height / scored.height):.3f}%)")
    print(f"  disagreeing     {mismatch.height} rows across "
          f"{mismatch['game_id'].n_unique()} games")
    if mismatch.height:
        print("\n  upstream pbp score columns are unreliable for these games; the")
        print("  schedule is authoritative and supplies the label, so the target")
        print("  is unaffected. By season:")
        print(mismatch.group_by("season").agg(games=pl.col("game_id").n_unique()).sort("season"))

    # -------------------------------------------- team_stats reconciliation
    header("7. RECONCILIATION vs nflverse load_team_stats (independent implementation)")
    ts = read_team_stats().select(
        "game_id", "team", "sacks_suffered", "passing_interceptions", "passing_cpoe"
    )
    joined = team_game.join(ts, on=["game_id", "team"], how="inner")
    print(f"  rows joined     {joined.height:,} of {team_game.height:,}")

    comparisons = [
        ("sacks taken", "off_sacks", "sacks_suffered"),
        ("interceptions thrown", "off_interceptions", "passing_interceptions"),
    ]
    print(f"\n  {'metric':<24}{'exact match':>14}{'rate':>10}{'max diff':>10}")
    for label, ours, theirs in comparisons:
        cmp = joined.filter(pl.col(ours).is_not_null() & pl.col(theirs).is_not_null())
        diff = (cmp[ours].cast(pl.Float64) - cmp[theirs].cast(pl.Float64)).abs()
        exact = (diff == 0).sum()
        print(f"  {label:<24}{exact:>14,}{exact / cmp.height:>9.2%}{diff.max():>10.0f}")

    cpoe = joined.filter(
        pl.col("off_cpoe").is_not_null() & pl.col("passing_cpoe").is_not_null()
    )
    cpoe_diff = (cpoe["off_cpoe"] - cpoe["passing_cpoe"]).abs()
    print(f"\n  cpoe: mean abs diff {cpoe_diff.mean():.4f}, "
          f"median {cpoe_diff.median():.4f}, max {cpoe_diff.max():.3f} "
          f"(n={cpoe.height:,})")
    print("  (small differences expected: we average cpoe over charted plays,")
    print("   nflverse averages over pass attempts)")

    # ------------------------------------------------------------- nulls
    header("8. NULL COUNTS BY ERA")
    early = team_game.filter(pl.col("season") < MATRIX_START_SEASON)
    late = team_game.filter(pl.col("season") >= MATRIX_START_SEASON)

    stat_cols = [f"off_{s}" for s in OFFENSE_STATS] + [f"def_{s}" for s in OFFENSE_STATS]
    rows = []
    for col in stat_cols:
        e, ln = early[col].null_count(), late[col].null_count()
        if e or ln:
            rows.append({
                "column": col,
                f"pre_{MATRIX_START_SEASON}": e,
                "pre_pct": round(100 * e / early.height, 2),
                f"{MATRIX_START_SEASON}_plus": ln,
                "plus_pct": round(100 * ln / late.height, 3),
            })
    if rows:
        print(pl.DataFrame(rows).sort("plus_pct", descending=True))
    else:
        print("  no nulls in any statistic column")
    print(f"\n  (rows: pre={early.height:,}  matrix={late.height:,})")

    # -------------------------------------------------------- hand checks
    header("9. HAND-CHECK GAMES (compare against public box scores)")
    def one(frame: pl.DataFrame, *, newest: bool = False) -> list[str]:
        """First (or last) game_id matching a filter, or empty if none."""
        return frame.sort("gameday", descending=newest)["game_id"].head(1).to_list()

    # Looked up by attribute rather than hardcoded: the designated home team in
    # a Super Bowl alternates by conference, so the game_id is not guessable.
    sb_2003 = one(schedules.filter((pl.col("season") == 2003) & (pl.col("game_type") == "SB")))
    sb_2016 = one(schedules.filter((pl.col("season") == 2016) & (pl.col("game_type") == "SB")))
    dome_game = one(schedules.filter((pl.col("season") == 2024) & (pl.col("roof") == "dome")))
    recent = one(
        schedules.filter((pl.col("season") == 2025) & pl.col("home_score").is_not_null()),
        newest=True,
    )

    targets = [
        *[("Super Bowl XXXVIII, 2003 (pre-2006: cpoe must be null)", g) for g in sb_2003],
        *[("Super Bowl LI, 2016 (overtime, neutral site)", g) for g in sb_2016],
        *[("2024 dome game (temp/wind must be null)", g) for g in dome_game],
        *[("most recent completed game", g) for g in recent],
    ]

    for label, gid in targets:
        rows_ = team_game.filter(pl.col("game_id") == gid)
        if rows_.height == 0:
            print(f"\n  {label}: {gid} NOT FOUND")
            continue
        sched_row = schedules.filter(pl.col("game_id") == gid)
        print(f"\n  {label}")
        print(f"  {gid}   roof={sched_row['roof'][0]}  temp={sched_row['temp'][0]}  "
              f"wind={sched_row['wind'][0]}  spread={sched_row['spread_line'][0]}")
        print(rows_.select(
            "team", "opponent", "is_home", "is_neutral_site",
            "points_for", "points_against", "won",
            "off_plays", "off_epa_per_play", "off_success_rate", "off_cpoe",
        ))


if __name__ == "__main__":
    main()
