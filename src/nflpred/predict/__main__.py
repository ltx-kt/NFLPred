"""The ``python -m nflpred.predict`` command line: predict, settle, report."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nflpred.config import (
    LIVE_SEASON,
    PICK_THRESHOLD,
)
from nflpred.features.pbp_agg import outcome
from nflpred.ingest import read_schedules
from nflpred.predict.run import frame_market_probability, log_week, predict_week
from nflpred.predict.week import WeekView, week_frame
from nflpred.predlog import (
    config_suffixes,
    connect,
    rolling_brier,
    settle,
    settled_predictions,
    summary,
    sync_schedule,
)

# ------------------------------------------------------------------- cli


def _picks_table(records: Sequence[dict[str, Any]], view: WeekView) -> pl.DataFrame:
    market = frame_market_probability(view.target)
    return pl.DataFrame(
        [
            {
                "game": f"{record['away_team']} @ {record['home_team']}",
                "pick": record["ensemble"]["pick"],
                "p(home)": record["ensemble"]["home_win_prob"],
                "confidence": record["ensemble"]["confidence"],
                "agreement": record["agreement"],
                "elo": round(float(view.target["elo_prob"][position]), 4),
                "market": None if np.isnan(market[position]) else round(float(market[position]), 4),
                "result": None
                if view.target["home_win"][position] is None
                else float(view.target["home_win"][position]),
            }
            for position, record in enumerate(records)
        ]
    )


def _run_week(args: argparse.Namespace) -> None:
    view = week_frame(args.season, args.week)
    records, fit, _ = predict_week(view, explain=args.explain)

    state = f"{len(view.synthesized)} unplayed" if view.live else "all completed"
    half_life = "off (unweighted)" if fit.half_life is None else f"{fit.half_life:g}"
    print(
        f"{view.season} week {view.week}: {view.target.height} games ({state}), "
        f"first kickoff {view.kickoff}"
    )
    print(
        f"  fitted on {fit.train.height:,} games "
        f"({'-'.join(str(s) for s in fit.seasons('train'))}), calibrated on "
        f"{fit.calib.height:,} ({'-'.join(str(s) for s in fit.seasons('calib'))}), "
        f"recency half-life {half_life}, {fit.seconds:.1f}s"
    )
    print(_picks_table(records, view))

    if args.explain:
        print("\nnarratives:")
        for record in records:
            print(f"  {record['away_team']} @ {record['home_team']}")
            print(f"    {record['explanation']['narrative']}")

    if not args.no_log:
        version = log_week(view, fit, records)
        print(f"  logged {len(records)} predictions as {version}")

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"  wrote {path}")


def _settle(_: argparse.Namespace) -> None:
    schedules = read_schedules()
    results = schedules.filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    ).select(
        "game_id",
        "home_score",
        "away_score",
        home_win=outcome(pl.col("home_score") - pl.col("away_score")),
    )

    connection = connect()
    try:
        written = settle(connection, results)
        synced = sync_schedule(connection, schedules, season=LIVE_SEASON)
        counts = summary(connection)
    finally:
        connection.close()

    print(f"settled {written} outcomes onto logged predictions")
    print(
        f"  {counts['settled']} of {counts['predictions']} logged predictions now have "
        f"a result ({counts['settled_share']:.1%})"
    )
    print(f"synced {synced} games of {LIVE_SEASON}'s schedule for the dashboard")


def _report(args: argparse.Namespace) -> None:
    connection = connect()
    try:
        counts = summary(connection)
        suffixes = config_suffixes(connection)
        current = args.config or (suffixes[0] if suffixes else None)
        overall, trailing = rolling_brier(connection, window=args.window, suffix=current)
        scored = settled_predictions(connection, suffix=current)
    finally:
        connection.close()

    print(
        f"prediction log: {counts['runs']} runs, {counts['predictions']} predictions, "
        f"{counts['outcomes']} outcomes, {counts['settled_share']:.1%} settled"
    )
    print(f"  configurations in the log: {', '.join(suffixes) or 'none'}")
    print(
        f"  reporting on {current} only - pooling two configurations' predictions on "
        f"the same games would score a model that never existed."
    )
    if overall.is_empty():
        print(
            "  nothing settled for this configuration yet - run "
            "`python -m nflpred.predict --settle`, or pass --config to pick another."
        )
        return

    print(f"\nper-member Brier, all settled games ({scored.height} games):")
    print(overall)
    print(f"\nper-member Brier, trailing {args.window} weeks:")
    print(trailing)
    if trailing["n"].to_list() == overall["n"].to_list():
        print(
            f"  identical to the table above: this configuration has at most "
            f"{args.window} settled weeks, so the trailing window covers all of them."
        )
    print(
        "  Reported, never fed back into the ensemble weights - the members correlate "
        "at 0.959 and every weighting scheme lands within 0.0006 of every other, so\n"
        "  re-deriving weights weekly from a handful of games would be fitting noise "
        "on a flat surface. This is a monitoring signal (spec in-season item 3)."
    )

    ensemble = scored.select(
        squared_error=(pl.col("home_win_prob") - pl.col("home_win")) ** 2,
        correct=(
            (pl.col("home_win_prob") >= PICK_THRESHOLD).cast(pl.Float64) - pl.col("home_win")
        ).abs(),
    )
    print(
        f"\nensemble over the same games: brier "
        f"{ensemble['squared_error'].mean():.4f}, accuracy "
        f"{1.0 - ensemble['correct'].mean():.4f}"
    )


def main() -> None:
    # Windows consoles default to cp1252, which cannot encode the picks table's
    # box-drawing characters or the em dash in warnings; without this the weekly
    # run raises UnicodeEncodeError mid-print, before the log write. No-op where
    # stdout is already UTF-8 or is not a real stream (a pipe under a test).
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        prog="python -m nflpred.predict", description="Predict one week, or read the log."
    )
    parser.add_argument("--season", type=int, help="season to predict")
    parser.add_argument("--week", type=int, help="week to predict")
    parser.add_argument(
        "--explain",
        action="store_true",
        help="attach the Phase 6 explanation to every record",
    )
    parser.add_argument("--json", help="write the records to this path")
    parser.add_argument("--no-log", action="store_true", help="do not write to the prediction log")
    parser.add_argument(
        "--settle",
        action="store_true",
        help="join completed results onto logged predictions",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="per-member rolling Brier, computed from the log alone",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=4,
        help="how many trailing weeks --report's rolling window covers (default 4)",
    )
    parser.add_argument(
        "--config",
        help="config suffix for --report to score (default: the most recent)",
    )
    args = parser.parse_args()

    if args.settle:
        _settle(args)
        return
    if args.report:
        _report(args)
        return
    if args.season is None or args.week is None:
        parser.error("--season and --week are required unless --settle or --report")

    _run_week(args)


if __name__ == "__main__":
    main()
