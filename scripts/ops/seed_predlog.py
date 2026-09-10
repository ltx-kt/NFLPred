"""Backfill the prediction log by replaying completed weeks through the live path.

The log (`data/predictions.sqlite`) only holds the handful of weeks that have
been run by hand. A history browser and a calibration curve need more than that,
so this script walks a range of completed seasons and, for each week, makes the
**same three calls** `nflpred.predict._run_week` makes:

    week_frame(season, week)  ->  predict_week(view)  ->  log_week(view, fit, records)

Deliberately not a second writer built on `nflpred.backtest.walk_forward`: routing
the seed through `nflpred.predict` guarantees a seeded row is byte-for-byte the shape
of a live one, and keeps the log's single-writer property (D-28).

Two things to know about what this writes:

* **`--explain` is off.** SHAP roughly triples the per-week cost and the
  templated narrative is for an upcoming game, not a 2023 backtest. Seeded rows
  carry no `record["explanation"]`; everything downstream treats it as optional.
* **The config suffix moves.** `predlog.config_suffix` hashes the train/calib
  season *bounds*, which slide forward as the walk advances, so weeks from
  different seasons land under different `model_version` suffixes. That is
  correct - each is a genuinely different fit - but it means the seeded range is
  a **backtest of the spent test split (D-1)**, not one comparable model, and
  the UI labels it as such.

Run::

    uv run python scripts/seed_predlog.py                 # 2022-2025, all weeks
    uv run python scripts/seed_predlog.py --seasons 2024 2025
    uv run python scripts/seed_predlog.py --limit 1 --no-log   # time one week
"""

from __future__ import annotations

import argparse
import sys
import time

import polars as pl

from nflpred.ingest import read_pbp, read_schedules
from nflpred.predict import (
    TEAM_GAME_GT_PATH,
    connect,
    log_week,
    predict_week,
    summary,
    week_frame,
)


def completed_weeks(schedules: pl.DataFrame, first: int, last: int) -> list[tuple[int, int]]:
    """Every (season, week) in the range that has at least one finished game.

    Derived from the schedule rather than guessed as ``range(1, 23)`` so playoff
    weeks and the pre-2021 17-week seasons fall out for free, and a week with no
    results (a mid-seed live season) is simply absent.
    """
    played = (
        schedules.filter(
            pl.col("season").is_between(first, last)
            & pl.col("home_score").is_not_null()
            & pl.col("away_score").is_not_null()
        )
        .select("season", "week")
        .unique()
        .sort("season", "week")
    )
    return [(int(r["season"]), int(r["week"])) for r in played.iter_rows(named=True)]


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        type=int,
        nargs=2,
        metavar=("FIRST", "LAST"),
        default=(2022, 2025),
        help="inclusive season range to replay (default: 2022 2025)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stop after this many weeks (for timing a single fit)",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="predict but do not write to the log (dry run)",
    )
    args = parser.parse_args()
    first, last = args.seasons

    if not TEAM_GAME_GT_PATH.exists():
        parser.error(
            f"{TEAM_GAME_GT_PATH.name} is not built. Run `uv run python -m "
            f"nflpred.features.build` first - the seed slices the built matrix per week."
        )

    # Read the three big inputs once and thread them through every week; the
    # default `week_frame` re-reads all three from parquet on each call.
    print("loading team-game table, schedules, and play-by-play ...", flush=True)
    team_game = pl.read_parquet(TEAM_GAME_GT_PATH)
    schedules = read_schedules()
    pbp = read_pbp()

    targets = completed_weeks(schedules, first, last)
    if args.limit is not None:
        targets = targets[: args.limit]
    print(f"{len(targets)} completed weeks in {first}-{last} to replay\n", flush=True)

    started = time.perf_counter()
    written = 0
    skipped: list[str] = []

    for position, (season, week) in enumerate(targets, start=1):
        tag = f"{season} w{week:02d}"
        try:
            view = week_frame(season, week, team_game=team_game, schedules=schedules, pbp=pbp)
        except ValueError as error:
            skipped.append(f"{tag}: {error}")
            print(f"  {tag}  SKIP  {error}", flush=True)
            continue

        if view.live:
            skipped.append(f"{tag}: {len(view.synthesized)} games still unplayed")
            print(f"  {tag}  SKIP  week is not fully played", flush=True)
            continue

        records, fit, _ = predict_week(view, explain=False)
        elapsed = time.perf_counter() - started
        if args.no_log:
            print(
                f"  {tag}  {view.target.height:2d} games  fit {fit.seconds:5.1f}s  "
                f"[{position}/{len(targets)}]  dry run  ({elapsed:5.1f}s total)",
                flush=True,
            )
            continue

        version = log_week(view, fit, records)
        written += len(records)
        print(
            f"  {tag}  {view.target.height:2d} games  fit {fit.seconds:5.1f}s  "
            f"logged as {version}  [{position}/{len(targets)}]  ({elapsed:5.1f}s total)",
            flush=True,
        )

    total = time.perf_counter() - started
    print(f"\ndone in {total:.1f}s: {written} predictions written", flush=True)
    if skipped:
        print(f"{len(skipped)} weeks skipped:")
        for line in skipped:
            print(f"  {line}")

    if not args.no_log:
        connection = connect()
        try:
            counts = summary(connection)
        finally:
            connection.close()
        print(
            f"\nlog now holds {counts['runs']} runs, {counts['predictions']} predictions, "
            f"{counts['outcomes']} outcomes ({counts['settled_share']:.1%} settled)"
        )
        print("next: `uv run python -m nflpred.predict --settle` to attach outcomes")


if __name__ == "__main__":
    main()
