"""The prediction log's read helpers, on the states that used to break them.

`predlog._frame` returns a zero-column frame when a query is empty, so any
caller that filters its result on a named column crashes on a log that has
predictions written but nothing settled yet. `predlog.rolling_brier` guards
against this; `nflpred.predict._report` used to not, and died with a Polars
`ColumnNotFoundError` before it could print its "nothing settled" message.

These tests build a log with SQL rather than through the live path, so they
carry no `data` marker and need no built matrix.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import UTC, datetime

from nflpred import predict
from nflpred.predlog import (
    connect,
    record_run,
    settled_predictions,
    suffix_of,
)

SUFFIX = "abc123"
VERSION = f"2025w08-{SUFFIX}"


def _log_with_unsettled_predictions(path) -> sqlite3.Connection:
    """A log holding one run and two predictions, none of them settled."""
    connection = connect(path)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    connection.execute(
        """
        INSERT INTO runs (
            model_version, generated_at, season, week, league_week,
            train_first, train_last, n_train, calib_first, calib_last, n_calib,
            half_life, features, matrix_sha256_16, versions, meta_coefficients
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (VERSION, now, 2025, 8, 500, 2006, 2015, 2670, 2016, 2018, 801,
         60.0, json.dumps(["elo_prob"]), "0" * 16, json.dumps({"python": "3.13"}), None),
    )
    for i, game_id in enumerate(("2025_08_AAA_BBB", "2025_08_CCC_DDD")):
        connection.execute(
            """
            INSERT INTO predictions (
                game_id, model_version, generated_at, season, week, gameday,
                home_team, away_team, feature_hash, home_win_prob, pick,
                confidence, agreement, elo_prob, market_prob, record
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (game_id, VERSION, now, 2025, 8, "2025-10-26",
             game_id[-3:], game_id[8:11], "f" * 16, 0.6 + 0.01 * i, game_id[-3:],
             "moderate", "5/6", 0.58, 0.55, json.dumps({"game_id": game_id})),
        )
    connection.commit()
    return connection


def test_suffix_of_splits_on_the_last_hyphen() -> None:
    assert suffix_of("2025w08-abc123") == "abc123"
    assert suffix_of("2025w08-3f9c1a") == "3f9c1a"


def test_settled_predictions_on_an_empty_log_is_an_empty_frame(tmp_path) -> None:
    connection = connect(tmp_path / "predictions.sqlite")
    try:
        assert settled_predictions(connection).is_empty()
        # The suffix filter must not raise on the zero-column empty frame.
        assert settled_predictions(connection, suffix=SUFFIX).is_empty()
    finally:
        connection.close()


def test_settled_predictions_suffix_filter_survives_unsettled_predictions(tmp_path) -> None:
    connection = _log_with_unsettled_predictions(tmp_path / "predictions.sqlite")
    try:
        # Predictions exist, but the JOIN to outcomes yields nothing, so the
        # frame is still zero-column. This filter used to crash _report.
        result = settled_predictions(connection, suffix=SUFFIX)
        assert result.is_empty()
    finally:
        connection.close()


def test_record_run_refreshes_provenance_on_a_rerun(tmp_path) -> None:
    """A second write for the same model_version must not keep stale row counts.

    `config_suffix` hashes only the season bounds, half-life, features and
    library versions - not the matrix hash or the counts - so a rerun after the
    matrix is rebuilt reuses the version. The upsert has to carry the new
    provenance through, the way `record_predictions` does.
    """
    connection = connect(tmp_path / "predictions.sqlite")
    common = dict(
        season=2025, week=8, league_week=500, train_seasons=(2006, 2015),
        calib_seasons=(2016, 2018), half_life=60.0, features=["elo_prob"],
        versions={"python": "3.13"}, meta_coefficients=None,
    )
    try:
        record_run(connection, VERSION, n_train=2670, n_calib=801,
                   matrix_sha256_16="a" * 16, **common)
        record_run(connection, VERSION, n_train=9999, n_calib=1234,
                   matrix_sha256_16="b" * 16, **common)
        row = connection.execute(
            "SELECT n_train, n_calib, matrix_sha256_16 FROM runs WHERE model_version = ?",
            (VERSION,),
        ).fetchone()
        assert (row["n_train"], row["n_calib"], row["matrix_sha256_16"]) == (
            9999, 1234, "b" * 16
        )
    finally:
        connection.close()


def test_report_on_an_unsettled_log_prints_and_does_not_crash(
    tmp_path, monkeypatch, capsys
) -> None:
    path = tmp_path / "predictions.sqlite"
    _log_with_unsettled_predictions(path).close()
    monkeypatch.setattr(predict, "connect", lambda: connect(path))

    predict._report(argparse.Namespace(config=None, window=4))

    out = capsys.readouterr().out
    assert "nothing settled for this configuration yet" in out
