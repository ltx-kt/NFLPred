"""The read-only API, exercised over a throwaway log built by the live path.

The fixture runs two completed weeks through the same `week_frame` ->
`predict_week` -> `log_week` calls `nflpred.predict._run_week` makes, into a
temp-file log, then settles one of them. So every row the API reads here was
written exactly as a real weekly run writes it - no hand-built records - and the
tests can assert on the settled/unsettled split, the missing-explanation path,
and the 404s.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pytest

from nflpred.config import FRONTEND_DIST, TEAM_GAME_GT_PATH

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from nflpred.api.db import log_connection  # noqa: E402
from nflpred.api.main import app  # noqa: E402
from nflpred.ingest import read_pbp, read_schedules  # noqa: E402
from nflpred.predict import log_week, predict_week, week_frame  # noqa: E402
from nflpred.predlog import connect, settle  # noqa: E402

SETTLED_WEEK = (2023, 5)
UNSETTLED_WEEK = (2023, 6)


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    if not TEAM_GAME_GT_PATH.exists():
        pytest.skip("team_game_gt.parquet not built; run python -m nflpred.features.build")

    log_path = tmp_path_factory.mktemp("api") / "predictions.sqlite"
    schedules = read_schedules()
    team_game = pl.read_parquet(TEAM_GAME_GT_PATH)
    pbp = read_pbp()

    for season, week in (SETTLED_WEEK, UNSETTLED_WEEK):
        view = week_frame(season, week, team_game=team_game, schedules=schedules, pbp=pbp)
        records, fit, _ = predict_week(view, explain=False)
        log_week(view, fit, records, path=log_path)

    # Settle only the first week, so the API has both a scored and an unscored week.
    season, week = SETTLED_WEEK
    finished = schedules.filter(
        (pl.col("season") == season)
        & (pl.col("week") == week)
        & pl.col("home_score").is_not_null()
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
    connection = connect(log_path)
    try:
        settle(connection, finished)
    finally:
        connection.close()

    def _readonly() -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(
            f"file:{Path(log_path).as_posix()}?mode=ro", uri=True, check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    app.dependency_overrides[log_connection] = _readonly
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_counts_two_weeks(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["runs"] == 2
    assert body["predictions"] > 0
    assert 0.0 < body["settled_share"] < 1.0  # one week settled, one not


def test_index_lists_both_weeks(client: TestClient) -> None:
    body = client.get("/api/index").json()
    weeks = {(s["season"], w) for s in body["seasons"] for w in s["weeks"]}
    assert SETTLED_WEEK in weeks and UNSETTLED_WEEK in weeks
    assert body["latest"] is not None
    assert body["seasons"][0]["kind"] == "backtest"


def test_settled_week_has_outcomes(client: TestClient) -> None:
    body = client.get("/api/weeks/{}/{}".format(*SETTLED_WEEK)).json()
    assert body["ref"]["n_settled"] == body["ref"]["n_games"]
    for game in body["games"]:
        assert game["outcome"] is not None
        assert game["outcome"]["pick_correct"] in (0.0, 0.5, 1.0)
        assert len(game["members"]) == 6
        assert game["has_explanation"] is False  # seeded/backtest weeks carry none


def test_unsettled_week_outcome_is_null(client: TestClient) -> None:
    body = client.get("/api/weeks/{}/{}".format(*UNSETTLED_WEEK)).json()
    assert body["ref"]["n_settled"] == 0
    assert all(game["outcome"] is None for game in body["games"])


def test_game_detail_without_explanation_does_not_500(client: TestClient) -> None:
    week = client.get("/api/weeks/{}/{}".format(*SETTLED_WEEK)).json()
    game_id = week["games"][0]["game_id"]
    body = client.get(f"/api/games/{game_id}").json()
    assert "explanation" not in body["record"]
    assert body["record"]["game_id"] == game_id
    assert body["available_versions"] == [body["model_version"]]


def test_unknown_week_and_game_are_404(client: TestClient) -> None:
    assert client.get("/api/weeks/1999/1").status_code == 404
    assert client.get("/api/games/2023_05_NOPE_XXX").status_code == 404


def test_performance_is_within_the_leakage_tripwire(client: TestClient) -> None:
    body = client.get("/api/performance").json()
    assert body["n_settled"] > 0
    ensemble = next(row for row in body["overall"] if row["model"] == "ensemble")
    assert ensemble["accuracy"] <= body["accuracy_ceiling"]


def test_calibration_returns_points_or_empty(client: TestClient) -> None:
    body = client.get("/api/calibration?bins=5").json()
    assert body["bins"] == 5
    assert len(body["points"]) in (0, 5)


def test_members_scoped_to_one_config(client: TestClient) -> None:
    body = client.get("/api/members?window=4").json()
    assert body["window"] == 4
    assert {row["member"] for row in body["overall"]} <= {
        "logreg",
        "random forest",
        "xgboost",
        "lightgbm",
        "catboost",
        "elo",
    }


@pytest.mark.skipif(
    not (FRONTEND_DIST / "index.html").is_file(),
    reason="SPA catch-all is only mounted when frontend/dist is built",
)
def test_spa_catch_all_does_not_serve_files_outside_the_build_tree() -> None:
    """`%2e%2e` segments must not escape frontend/dist onto the repo root.

    `Path.is_relative_to` is lexical, so before the `.resolve()` guard a request
    for `/%2e%2e/%2e%2e/pyproject.toml` walked out of the build tree and returned
    the file. The route must fall back to index.html instead.
    """
    index_bytes = (FRONTEND_DIST / "index.html").read_bytes()
    with TestClient(app) as spa_client:
        resp = spa_client.get("/%2e%2e/%2e%2e/pyproject.toml")
    assert resp.status_code == 200
    assert resp.content == index_bytes
    assert b"[project]" not in resp.content
