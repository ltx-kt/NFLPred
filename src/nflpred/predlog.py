"""The prediction log: SQLite, four tables, and the reason it exists.

The spec's fourth in-season item, and the blunt version of why it is there:
*without this, there is no way to tell whether a change actually helped.* A
weekly system that only prints has no memory, and a model change three months
from now would be compared against a recollection.

SQLite rather than DuckDB (D-28). It is in the standard library, so the weekly
pipeline gains no dependency; the volume is ~272 rows a season; and the file
sits under the already-gitignored ``data/`` tree. Nothing here needs columnar
scans.

Four tables, and the split between them is the point:

``runs``
    One row per ``model_version`` — the *config*. Train and calibration ranges,
    row counts, half-life, feature tuple, matrix hash, library versions,
    meta-learner coefficients. D-30 leaves native model formats deferred and
    makes this row the reproducibility record instead: the harness refits per
    week and never reloads a pickle, so what has to survive is the recipe, not
    the object.
``predictions``
    One row per ``(game_id, model_version)``, upserted, so rerunning a week is
    idempotent rather than additive. Carries the flat fields a query wants and
    the full output-contract record as JSON alongside — the JSON is the
    deliverable, the columns are what makes the log answerable.
``member_predictions``
    Each member's probability, flat. The spec's in-season item 3 asks for a
    rolling Brier score per member; with the votes only inside the JSON that
    would be a walk over every record, and it is a ``GROUP BY`` instead.
``outcomes``
    Filled by ``--settle`` once games finish. Separate from ``predictions``
    because a prediction is immutable and a result arrives later; joining them
    is what lets the log score itself without ever rewriting what was predicted.

``model_version`` is ``{season}w{week:02d}-{suffix}``, matching the spec's
``2025w08-a`` shape, where the suffix is a short hash of the configuration.
Same config, same version; changed config, different version. That is the
property the whole log rests on — two rows under one version must mean two
predictions from one model.
"""

from __future__ import annotations

import json
import platform
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nflpred.config import PREDICTIONS_DB

#: Length of the configuration hash in ``model_version``. Six hex characters is
#: 16.7M configurations — ample for distinguishing the handful a project has,
#: and short enough that a version string stays readable in a table.
SUFFIX_LENGTH: int = 6

SCHEMA: str = """
CREATE TABLE IF NOT EXISTS runs (
    model_version    TEXT PRIMARY KEY,
    generated_at     TEXT NOT NULL,
    season           INTEGER NOT NULL,
    week             INTEGER NOT NULL,
    league_week      INTEGER,
    train_first      INTEGER,
    train_last       INTEGER,
    n_train          INTEGER,
    calib_first      INTEGER,
    calib_last       INTEGER,
    n_calib          INTEGER,
    half_life        REAL,
    features         TEXT NOT NULL,
    matrix_sha256_16 TEXT,
    versions         TEXT NOT NULL,
    meta_coefficients TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    game_id       TEXT NOT NULL,
    model_version TEXT NOT NULL,
    generated_at  TEXT NOT NULL,
    season        INTEGER NOT NULL,
    week          INTEGER NOT NULL,
    gameday       TEXT,
    home_team     TEXT NOT NULL,
    away_team     TEXT NOT NULL,
    feature_hash  TEXT NOT NULL,
    home_win_prob REAL NOT NULL,
    pick          TEXT NOT NULL,
    confidence    TEXT NOT NULL,
    agreement     TEXT NOT NULL,
    elo_prob      REAL,
    market_prob   REAL,
    record        TEXT NOT NULL,
    PRIMARY KEY (game_id, model_version),
    FOREIGN KEY (model_version) REFERENCES runs (model_version)
);

CREATE TABLE IF NOT EXISTS member_predictions (
    game_id       TEXT NOT NULL,
    model_version TEXT NOT NULL,
    member        TEXT NOT NULL,
    home_win_prob REAL NOT NULL,
    pick          TEXT NOT NULL,
    PRIMARY KEY (game_id, model_version, member)
);

CREATE TABLE IF NOT EXISTS outcomes (
    game_id    TEXT PRIMARY KEY,
    settled_at TEXT NOT NULL,
    home_score INTEGER,
    away_score INTEGER,
    home_win   REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS predictions_by_week
    ON predictions (season, week);
CREATE INDEX IF NOT EXISTS member_predictions_by_member
    ON member_predictions (member);
"""


def connect(path: Path = PREDICTIONS_DB) -> sqlite3.Connection:
    """Open the log, creating the file and the schema if they do not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    return connection


# ------------------------------------------------------------- identity


def library_versions() -> dict[str, str]:
    """Library versions that can move a prediction.

    The same set `nflpred.modeling.store` records, for the same reason and with the
    same caveat: recorded, not enforced. A mismatch is context for a number that
    moved, not a failure on its own.
    """
    import catboost
    import lightgbm
    import sklearn
    import xgboost

    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "lightgbm": lightgbm.__version__,
        "catboost": catboost.__version__,
    }


def config_suffix(
    train_seasons: tuple[int, int],
    calib_seasons: tuple[int, int],
    half_life: float | None,
    features: Sequence[str],
    versions: Mapping[str, str],
) -> str:
    """Short hash of everything that decides what the model is.

    Deliberately **not** a hash of the fitted objects. Two runs of the same
    recipe produce bit-identical models here (`tests/test_models.py` pins that),
    so hashing the recipe answers the question a version string is asked — "is
    this the same model?" — while staying computable before the fit and
    readable afterwards.
    """
    payload = json.dumps(
        {
            "train": list(train_seasons),
            "calib": list(calib_seasons),
            "half_life": half_life,
            "features": list(features),
            "versions": dict(sorted(versions.items())),
        },
        sort_keys=True,
    )
    return sha256(payload.encode()).hexdigest()[:SUFFIX_LENGTH]


def model_version(season: int, week: int, suffix: str) -> str:
    """``2025w08-3f9c1a`` — the spec's shape, with a config hash for a suffix."""
    return f"{season}w{week:02d}-{suffix}"


def feature_hash(values: Sequence[float]) -> str:
    """sha256 of one game's ordered feature vector.

    The spec asks predictions to be persisted with a feature-vector hash. Order
    matters and is the feature tuple's order, so this changes if the columns are
    reordered — which is the point: the same numbers under a different layout
    are a different input.
    """
    payload = ",".join(f"{float(v):.10g}" for v in values)
    return sha256(payload.encode()).hexdigest()[:16]


# ------------------------------------------------------------- writing


def record_run(
    connection: sqlite3.Connection,
    version: str,
    *,
    season: int,
    week: int,
    league_week: int | None,
    train_seasons: tuple[int, int],
    n_train: int,
    calib_seasons: tuple[int, int],
    n_calib: int,
    half_life: float | None,
    features: Sequence[str],
    matrix_sha256_16: str | None,
    versions: Mapping[str, str],
    meta_coefficients: Mapping[str, float] | None,
) -> None:
    """Upsert the configuration row. Rerunning a week rewrites it in place."""
    connection.execute(
        """
        INSERT INTO runs (
            model_version, generated_at, season, week, league_week,
            train_first, train_last, n_train,
            calib_first, calib_last, n_calib,
            half_life, features, matrix_sha256_16, versions, meta_coefficients
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT (model_version) DO UPDATE SET
            generated_at = excluded.generated_at,
            meta_coefficients = excluded.meta_coefficients
        """,
        (
            version,
            datetime.now(UTC).isoformat(timespec="seconds"),
            season,
            week,
            league_week,
            train_seasons[0],
            train_seasons[1],
            n_train,
            calib_seasons[0],
            calib_seasons[1],
            n_calib,
            half_life,
            json.dumps(list(features)),
            matrix_sha256_16,
            json.dumps(dict(sorted(versions.items()))),
            json.dumps(dict(meta_coefficients)) if meta_coefficients else None,
        ),
    )


def record_predictions(
    connection: sqlite3.Connection,
    version: str,
    records: Sequence[Mapping[str, Any]],
    frame: pl.DataFrame,
    features: Sequence[str],
    market: np.ndarray | None = None,
) -> int:
    """Upsert one week of predictions and their per-member votes.

    ``records`` are output-contract records in ``frame``'s row order —
    :func:`~nflpred.modeling.ensemble.predict_records` output, with or without Phase
    6's ``explanation`` key merged on. Idempotent by ``(game_id,
    model_version)``: a rerun of the same week under the same config replaces
    its own rows rather than appending a second opinion.
    """
    x = frame.select(features).to_numpy().astype(float)
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    elo = frame["elo_prob"].to_numpy().astype(float)

    rows = []
    member_rows = []
    for position, record in enumerate(records):
        ensemble = record["ensemble"]
        quoted = None if market is None else float(market[position])
        rows.append(
            (
                record["game_id"],
                version,
                generated_at,
                int(frame["season"][position]),
                int(frame["week"][position]),
                str(frame["gameday"][position]),
                record["home_team"],
                record["away_team"],
                feature_hash(x[position]),
                float(ensemble["home_win_prob"]),
                ensemble["pick"],
                ensemble["confidence"],
                record["agreement"],
                float(elo[position]),
                None if quoted is None or np.isnan(quoted) else quoted,
                json.dumps(record),
            )
        )
        for member, vote in record["model_votes"].items():
            member_rows.append(
                (
                    record["game_id"],
                    version,
                    member,
                    float(vote["home_win_prob"]),
                    vote["pick"],
                )
            )

    connection.executemany(
        """
        INSERT INTO predictions (
            game_id, model_version, generated_at, season, week, gameday,
            home_team, away_team, feature_hash, home_win_prob, pick,
            confidence, agreement, elo_prob, market_prob, record
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT (game_id, model_version) DO UPDATE SET
            generated_at  = excluded.generated_at,
            feature_hash  = excluded.feature_hash,
            home_win_prob = excluded.home_win_prob,
            pick          = excluded.pick,
            confidence    = excluded.confidence,
            agreement     = excluded.agreement,
            elo_prob      = excluded.elo_prob,
            market_prob   = excluded.market_prob,
            record        = excluded.record
        """,
        rows,
    )
    connection.executemany(
        """
        INSERT INTO member_predictions (game_id, model_version, member, home_win_prob, pick)
        VALUES (?,?,?,?,?)
        ON CONFLICT (game_id, model_version, member) DO UPDATE SET
            home_win_prob = excluded.home_win_prob,
            pick          = excluded.pick
        """,
        member_rows,
    )
    connection.commit()
    return len(rows)


def settle(connection: sqlite3.Connection, results: pl.DataFrame) -> int:
    """Join completed results onto logged predictions.

    ``results`` needs ``game_id``, ``home_score``, ``away_score`` and
    ``home_win`` — the last as the D-4 float, so a tie settles as 0.5 here
    exactly as it scores as 0.5 everywhere else rather than being rounded into a
    win by the log.

    Only games that were actually predicted are written: an outcome for a game
    the system never had an opinion on is not evidence about the system.
    """
    predicted = {
        row["game_id"]
        for row in connection.execute("SELECT DISTINCT game_id FROM predictions")
    }
    settled_at = datetime.now(UTC).isoformat(timespec="seconds")

    rows = [
        (
            row["game_id"],
            settled_at,
            None if row["home_score"] is None else int(row["home_score"]),
            None if row["away_score"] is None else int(row["away_score"]),
            float(row["home_win"]),
        )
        for row in results.iter_rows(named=True)
        if row["game_id"] in predicted and row["home_win"] is not None
    ]

    connection.executemany(
        """
        INSERT INTO outcomes (game_id, settled_at, home_score, away_score, home_win)
        VALUES (?,?,?,?,?)
        ON CONFLICT (game_id) DO UPDATE SET
            settled_at = excluded.settled_at,
            home_score = excluded.home_score,
            away_score = excluded.away_score,
            home_win   = excluded.home_win
        """,
        rows,
    )
    connection.commit()
    return len(rows)


# ------------------------------------------------------------- reading


def _frame(connection: sqlite3.Connection, sql: str) -> pl.DataFrame:
    rows = [dict(row) for row in connection.execute(sql)]
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def settled_predictions(connection: sqlite3.Connection) -> pl.DataFrame:
    """Every logged ensemble prediction that has an outcome, one row per game."""
    return _frame(
        connection,
        """
        SELECT p.game_id, p.model_version, p.season, p.week, p.gameday,
               p.home_team, p.away_team, p.home_win_prob, p.pick,
               p.confidence, p.elo_prob, p.market_prob, o.home_win
        FROM predictions p
        JOIN outcomes o USING (game_id)
        ORDER BY p.gameday, p.game_id
        """,
    )


def settled_member_predictions(connection: sqlite3.Connection) -> pl.DataFrame:
    """Every logged per-member probability that has an outcome.

    The table the rolling-Brier report is computed from — a join and a group-by
    rather than a walk over JSON, which is the whole reason
    ``member_predictions`` is flat.
    """
    return _frame(
        connection,
        """
        SELECT m.game_id, m.model_version, m.member, m.home_win_prob,
               p.season, p.week, p.gameday, o.home_win
        FROM member_predictions m
        JOIN predictions p USING (game_id, model_version)
        JOIN outcomes o USING (game_id)
        ORDER BY p.gameday, p.game_id, m.member
        """,
    )


def summary(connection: sqlite3.Connection) -> dict[str, Any]:
    """Counts a checkpoint can print: rows, versions, settled fraction."""
    def scalar(sql: str) -> int:
        return int(next(iter(connection.execute(sql)))[0])

    predictions = scalar("SELECT COUNT(*) FROM predictions")
    settled = scalar(
        "SELECT COUNT(*) FROM predictions p JOIN outcomes o USING (game_id)"
    )
    return {
        "runs": scalar("SELECT COUNT(*) FROM runs"),
        "predictions": predictions,
        "member_predictions": scalar("SELECT COUNT(*) FROM member_predictions"),
        "outcomes": scalar("SELECT COUNT(*) FROM outcomes"),
        "settled": settled,
        "settled_share": round(settled / predictions, 4) if predictions else 0.0,
        "model_versions": [
            row["model_version"]
            for row in connection.execute(
                "SELECT model_version FROM runs ORDER BY season, week, model_version"
            )
        ],
    }


def config_suffixes(connection: sqlite3.Connection) -> list[str]:
    """Distinct configuration suffixes in the log, newest first.

    More than one means the log spans a model change — which is what it is for,
    and also why :func:`rolling_brier` must not pool them.
    """
    seen: list[str] = []
    for row in connection.execute(
        "SELECT model_version FROM runs ORDER BY generated_at DESC, season DESC, week DESC"
    ):
        suffix = row["model_version"].rsplit("-", 1)[-1]
        if suffix not in seen:
            seen.append(suffix)
    return seen


def rolling_brier(
    connection: sqlite3.Connection, window: int = 4, suffix: str | None = None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per-member Brier score, overall and over the trailing ``window`` weeks.

    The spec's in-season item 3, as its own open-questions section resolved it:
    **report** this, never feed it back into the ensemble weights. The members
    correlate at 0.959 (D-21) and every weighting scheme tried lands within
    0.0006 of every other, so a mechanism that re-derived weights weekly from
    ~16 games would be fitting noise on a flat surface. It is a monitoring
    signal — a member whose Brier drifts is worth looking at, not worth
    automatically down-weighting.

    **Scoped to one configuration.** ``suffix`` defaults to the most recently
    written one, because pooling two configurations' predictions on the same
    games would report a Brier score for a model that never existed — and the
    log is deliberately built to hold several, so this is the ordinary case
    rather than an edge one. Comparing two of them is a matter of calling this
    twice, which is exactly the comparison the log exists to make possible.

    Returns ``(overall, trailing)``. Ties enter as 0.5 and are squared against
    directly rather than expanded, which is the same number
    :func:`~nflpred.evaluate.brier` returns for them.
    """
    frame = settled_member_predictions(connection)
    if frame.is_empty():
        return pl.DataFrame(), pl.DataFrame()

    suffixes = config_suffixes(connection)
    chosen = suffix or (suffixes[0] if suffixes else None)
    if chosen is not None:
        frame = frame.filter(pl.col("model_version").str.ends_with(f"-{chosen}"))
    if frame.is_empty():
        return pl.DataFrame(), pl.DataFrame()

    scored = frame.with_columns(
        squared_error=(pl.col("home_win_prob") - pl.col("home_win")) ** 2
    )

    def aggregate(subset: pl.DataFrame) -> pl.DataFrame:
        return (
            subset.group_by("member")
            .agg(
                n=pl.len(),
                brier=pl.col("squared_error").mean().round(4),
                mean_prob=pl.col("home_win_prob").mean().round(4),
            )
            .sort("brier")
        )

    weeks = scored.select("season", "week").unique().sort("season", "week")
    recent = weeks.tail(window)
    trailing = scored.join(recent, on=["season", "week"], how="semi")

    return aggregate(scored), aggregate(trailing)
