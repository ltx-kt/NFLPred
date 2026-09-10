"""The weekly run: build one week, refit, predict, explain, log.

Everything before this module reads a matrix in which every game has already
been played. This is the module that does not — the 2026 season has a schedule
and no results, and a fixture that has not kicked off produces no row from
:func:`nflpred.features.pbp_agg._spine`, which filters to games with scores.

**The synthesis, and why it is one row per team (D-27).** Rather than widening
the spine to admit unplayed games — which would put null-result rows into every
table the project builds, including the ones the leak tests audit — the target
week's scheduled games are turned into team-game rows with every statistic and
``won`` null, concatenated onto the completed table, and pushed through the
**existing** ``build_rolling`` / ``build_qb`` / ``build_elo`` / ``build_features``
pipeline unchanged. Three properties make that safe, and each is checked rather
than asserted in prose:

* Every rolling expression is ``shift(1)``ed, and the synthesized row is the
  **last** row of its team's series, so no null ever enters a window — the row
  reads the same eight completed games it would have read anyway, and no other
  row can see it.
* Elo already yields pre-game ratings for scoreless games and skips the update
  (``src/nflpred/features/elo.py``), so the rating feature needs nothing added.
* The projected starter is the previous game's actual starter (D-8), never
  ``schedules.home_qb_id``, which is derived post-game and is null here.

`tests/test_predict.py` takes a **completed** week, rebuilds it as if unplayed,
and demands the feature row come back identical. That is train/serve skew
measured rather than argued — and it is how D-31's quarterback-form skew was
found, since the composite failed exactly that test before it was fixed.

**Documented limit.** Only a team's *next* unplayed week is buildable. Asking
for week 12 in August would need weeks 8-11 simulated to know what form each
team carries into it, and inventing that is a different project. It is refused
with a message naming the weeks in the way.

Run::

    python -m nflpred.predict --season 2026 --week 1 --explain
    python -m nflpred.predict --season 2025 --week 8 --json week8.json
    python -m nflpred.predict --settle
    python -m nflpred.predict --report
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nflpred.backtest import WeeklyFit, fit_week
from nflpred.config import (
    CALIB_WINDOW_SEASONS,
    FEATURE_MATRIX_GT_PATH,
    POSTSEASON_TYPES,
    RECENCY_HALF_LIFE,
    TEAM_GAME_GT_PATH,
)
from nflpred.evaluate import market_probability
from nflpred.features.build import build_features
from nflpred.ingest import read_pbp, read_schedules
from nflpred.modeling.base import home_win_probability
from nflpred.modeling.ensemble import ENSEMBLE_FEATURES, meta_coefficients, predict_records
from nflpred.predlog import (
    config_suffix,
    config_suffixes,
    connect,
    library_versions,
    model_version,
    record_predictions,
    record_run,
    rolling_brier,
    settle,
    settled_predictions,
    summary,
)

#: Columns :func:`_synthesized_rows` must fill itself. Everything else in the
#: team-game table is a statistic and is null for a game nobody has played.
_SPINE_COLUMNS: tuple[str, ...] = (
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
)


@dataclass(frozen=True)
class WeekView:
    """One target week, and the history it is entitled to see."""

    season: int
    week: int
    league_week: int
    kickoff: date
    target: pl.DataFrame
    completed: pl.DataFrame
    matrix: pl.DataFrame
    synthesized: tuple[str, ...]

    @property
    def live(self) -> bool:
        """True when at least one game of the week has not been played."""
        return bool(self.synthesized)


# ------------------------------------------------------------- synthesis


def _synthesized_rows(week_schedule: pl.DataFrame, template: pl.DataFrame) -> pl.DataFrame:
    """Two team-game rows per unplayed fixture, statistics all null.

    Mirrors :func:`nflpred.features.pbp_agg._spine`'s output shape rather than
    calling it, because that function's job is to *exclude* these games. The
    template supplies dtypes so the concatenation below is a plain vertical
    one — a diagonal concat would paper over a column this forgot to fill.
    """
    shared = (
        pl.col("game_id"),
        pl.col("season"),
        pl.col("week"),
        pl.col("game_type"),
        pl.col("gameday"),
    )
    neutral = (pl.col("location") == "Neutral").cast(pl.Int8)
    postseason = pl.col("game_type").is_in(POSTSEASON_TYPES).cast(pl.Int8)

    sides = [
        week_schedule.select(
            *shared,
            team=pl.col(f"{side}_team"),
            opponent=pl.col(f"{other}_team"),
            is_home=pl.lit(1 if side == "home" else 0, dtype=pl.Int8),
            is_neutral_site=neutral,
            is_postseason=postseason,
        )
        for side, other in (("home", "away"), ("away", "home"))
    ]

    rows = pl.concat(sides, how="vertical")
    unknown = [c for c in template.columns if c not in _SPINE_COLUMNS]
    return rows.with_columns(
        [pl.lit(None, dtype=template.schema[c]).alias(c) for c in unknown]
    ).select([pl.col(c).cast(template.schema[c]) for c in template.columns])


def _blocking_weeks(
    schedules: pl.DataFrame, season: int, week: int, kickoff: date, teams: Sequence[str]
) -> list[int]:
    """Earlier weeks of ``season`` that ``teams`` have not played yet.

    The documented limit, as a computation. A non-empty answer means the
    requested week is not any of these teams' *next* game, so the rolling
    windows behind it would have to be simulated rather than observed.
    """
    involved = pl.col("home_team").is_in(list(teams)) | pl.col("away_team").is_in(list(teams))
    pending = schedules.filter(
        (pl.col("season") == season)
        & (pl.col("week") < week)
        & involved
        & (pl.col("gameday").str.to_date(strict=False) < kickoff)
        & (pl.col("home_score").is_null() | pl.col("away_score").is_null())
    )
    return sorted(set(pending["week"].to_list()))


def week_frame(
    season: int,
    week: int,
    team_game: pl.DataFrame | None = None,
    schedules: pl.DataFrame | None = None,
    pbp: pl.DataFrame | None = None,
) -> WeekView:
    """The feature rows for one week, whether or not it has been played.

    A week with results is a slice of the built matrix. A week without them is
    synthesized first — but through the *same* ``build_features`` call, so there
    is one construction of a feature row in the project and not two. A week that
    is half played (a Thursday game done, the Sunday slate not) synthesizes only
    the games that are missing, which falls out of the same code.
    """
    team_game = pl.read_parquet(TEAM_GAME_GT_PATH) if team_game is None else team_game
    schedules = read_schedules() if schedules is None else schedules
    pbp = read_pbp() if pbp is None else pbp

    week_schedule = schedules.filter(
        (pl.col("season") == season) & (pl.col("week") == week)
    )
    if not week_schedule.height:
        msg = f"no games scheduled for {season} week {week}."
        raise ValueError(msg)

    kickoff = week_schedule["gameday"].str.to_date(strict=False).min()
    unplayed = week_schedule.filter(
        pl.col("home_score").is_null() | pl.col("away_score").is_null()
    )

    if unplayed.height:
        teams = sorted(set(unplayed["home_team"].to_list() + unplayed["away_team"].to_list()))
        blocking = _blocking_weeks(schedules, season, week, kickoff, teams)
        if blocking:
            msg = (
                f"{season} week {week} is not the next unplayed week for these teams: "
                f"weeks {blocking} are still unplayed and sit before it. Predicting it "
                f"would require simulating those weeks to know what form each team "
                f"carries in, which this system does not do. Predict week "
                f"{blocking[0]} instead."
            )
            raise ValueError(msg)
        team_game = pl.concat(
            [team_game, _synthesized_rows(unplayed, team_game)], how="vertical"
        )

    matrix = build_features(team_game, schedules, pbp)
    target = matrix.filter(
        (pl.col("season") == season) & (pl.col("week") == week)
    ).sort("gameday", "game_id")
    if not target.height:
        msg = (
            f"{season} week {week} produced no feature rows. The season is before "
            f"the matrix start, or the schedule and the team-game table disagree."
        )
        raise ValueError(msg)

    missing = [c for c in ENSEMBLE_FEATURES if target[c].null_count()]
    if missing:
        msg = (
            f"{season} week {week} has null features {missing}. Every rolling window "
            f"is shift(1)ed and the synthesized row is last in its team's series, so "
            f"a null here means that invariant broke — investigate before predicting."
        )
        raise ValueError(msg)

    completed = matrix.filter(
        (pl.col("gameday") < kickoff) & pl.col("home_win").is_not_null()
    ).sort("gameday", "game_id")

    return WeekView(
        season=season,
        week=week,
        league_week=int(target["league_week"][0]),
        kickoff=kickoff,
        target=target,
        completed=completed,
        matrix=matrix,
        synthesized=tuple(unplayed["game_id"].to_list()),
    )


# ------------------------------------------------------------- prediction


def _market(frame: pl.DataFrame) -> np.ndarray:
    return market_probability(
        frame["home_moneyline"].to_numpy(), frame["away_moneyline"].to_numpy()
    )


def _matrix_fingerprint(frame: pl.DataFrame) -> str:
    """Content hash of the rows a fit actually saw.

    `nflpred.modeling.store` hashes the matrix *file*; there is no file here, because
    the live path rebuilds in memory to admit an unplayed row. Hashing the
    frame's contents answers the same question the manifest's hash answers:
    were these predictions made from the data we think they were.
    """
    digest = sha256()
    digest.update(frame.hash_rows().to_numpy().tobytes())
    return digest.hexdigest()[:16]


def predict_week(
    view: WeekView,
    half_life: float | None = RECENCY_HALF_LIFE,
    calib_seasons: int = CALIB_WINDOW_SEASONS,
    explain: bool = False,
) -> tuple[list[dict[str, Any]], WeeklyFit, np.ndarray]:
    """Refit on everything before kickoff, then predict the week.

    The fit is kept in memory and returned rather than saved and reloaded.
    D-30 leaves native model formats deferred for exactly this reason: there is
    nothing to reload *to*. ``--explain`` needs the raw inner estimators and the
    stack's ``final_estimator_``, and `nflpred.modeling.store.load_models` could not
    supply them anyway — its fingerprint recompute requires a matrix that still
    contains the 2019-2021 validation rows, which a live 2026 matrix does not
    single out.
    """
    fit = fit_week(
        view.completed,
        view.league_week,
        view.season,
        view.week,
        ENSEMBLE_FEATURES,
        half_life,
        calib_seasons,
    )

    scored = view.target.with_columns(market_prob=pl.Series(_market(view.target)))
    probability = home_win_probability(fit.stack, scored, ENSEMBLE_FEATURES)
    records = predict_records(fit.stack, fit.members, scored)

    if explain:
        # The Phase 6 call sequence, unchanged: the explainers are built over
        # the **raw** inner estimators, which are the objects the meta-learner
        # consumed and therefore the objects to explain.
        from nflpred.explain import build_explainers, explained_records, meta_terms

        coefficients, intercept = meta_terms(fit.stack)
        explainers = build_explainers(fit.raw, float(fit.train["elo_prob"].mean()))
        records = explained_records(
            records, scored, explainers, coefficients, intercept, probability
        )

    return records, fit, probability


def log_week(
    view: WeekView,
    fit: WeeklyFit,
    records: Sequence[dict[str, Any]],
    path: Path | None = None,
) -> str:
    """Write one week's run and predictions, and return the model version."""
    versions = library_versions()
    train_seasons, calib_seasons = fit.seasons("train"), fit.seasons("calib")
    suffix = config_suffix(
        train_seasons, calib_seasons, fit.half_life, ENSEMBLE_FEATURES, versions
    )
    version = model_version(view.season, view.week, suffix)

    connection = connect(path) if path else connect()
    try:
        record_run(
            connection,
            version,
            season=view.season,
            week=view.week,
            league_week=view.league_week,
            train_seasons=train_seasons,
            n_train=fit.train.height,
            calib_seasons=calib_seasons,
            n_calib=fit.calib.height,
            half_life=fit.half_life,
            features=ENSEMBLE_FEATURES,
            matrix_sha256_16=_matrix_fingerprint(view.completed),
            versions=versions,
            meta_coefficients=meta_coefficients(fit.stack),
        )
        record_predictions(
            connection,
            version,
            records,
            view.target,
            ENSEMBLE_FEATURES,
            market=_market(view.target),
        )
    finally:
        connection.close()
    return version


# ------------------------------------------------------------------- cli


def _picks_table(records: Sequence[dict[str, Any]], view: WeekView) -> pl.DataFrame:
    market = _market(view.target)
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

    state = (
        f"{len(view.synthesized)} unplayed" if view.live else "all completed"
    )
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
        home_win=pl.when(pl.col("home_score") > pl.col("away_score"))
        .then(1.0)
        .when(pl.col("home_score") < pl.col("away_score"))
        .then(0.0)
        .otherwise(0.5),
    )

    connection = connect()
    try:
        written = settle(connection, results)
        counts = summary(connection)
    finally:
        connection.close()

    print(f"settled {written} outcomes onto logged predictions")
    print(
        f"  {counts['settled']} of {counts['predictions']} logged predictions now have "
        f"a result ({counts['settled_share']:.1%})"
    )


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
        f"  reporting on {current} only — pooling two configurations' predictions on "
        f"the same games would score a model that never existed."
    )
    if overall.is_empty():
        print(
            "  nothing settled for this configuration yet — run "
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
        "  Reported, never fed back into the ensemble weights — the members correlate "
        "at 0.959 and every weighting scheme lands within 0.0006 of every other, so\n"
        "  re-deriving weights weekly from a handful of games would be fitting noise "
        "on a flat surface. This is a monitoring signal (spec in-season item 3)."
    )

    ensemble = scored.select(
        squared_error=(pl.col("home_win_prob") - pl.col("home_win")) ** 2,
        correct=((pl.col("home_win_prob") >= 0.5).cast(pl.Float64) - pl.col("home_win")).abs(),
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

    parser = argparse.ArgumentParser(description="Predict one week, or read the log.")
    parser.add_argument("--season", type=int, help="season to predict")
    parser.add_argument("--week", type=int, help="week to predict")
    parser.add_argument(
        "--explain",
        action="store_true",
        help="attach the Phase 6 explanation to every record",
    )
    parser.add_argument("--json", help="write the records to this path")
    parser.add_argument(
        "--no-log", action="store_true", help="do not write to the prediction log"
    )
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

    if not FEATURE_MATRIX_GT_PATH.exists():
        print(
            f"note: {FEATURE_MATRIX_GT_PATH.name} is not built. The week is rebuilt "
            f"from the team-game table either way; build it for the checkpoints.",
            file=sys.stderr,
        )
    _run_week(args)


if __name__ == "__main__":
    main()
