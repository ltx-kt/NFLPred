"""Reads the routes need that `nflpred.predlog` does not already expose.

`predlog` has the frames the scoring report is built from (`settled_predictions`,
`settled_member_predictions`, `summary`, `config_suffixes`). The board and the
game page need a different cut - one week's predictions joined to their member
votes and, where they exist, their outcomes - so those SELECTs live here. Same
four tables, same read-only connection.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from nflpred.config import LIVE_SEASON
from nflpred.predlog import config_suffixes, suffix_of

#: One game row joined to its outcome. The WHERE clause is the only thing that
#: differs between the board query and the single-game query.
_GAME_SQL = (
    "SELECT p.*, o.home_score, o.away_score, o.home_win "
    "FROM predictions p LEFT JOIN outcomes o USING (game_id) "
)


def _pick_correct(pick: str, home_team: str, home_win: float | None) -> float | None:
    """1.0 hit, 0.0 miss, 0.5 tie; None when the outcome is unknown.

    ``pick`` is a team code and a home pick is ``pick == home_team``; an away
    pick is anything else, so the away code is not needed here.
    """
    if home_win is None:
        return None
    picked_home = pick == home_team
    if home_win == 0.5:
        return 0.5
    home_won = home_win == 1.0
    return 1.0 if picked_home == home_won else 0.0


def index(connection: sqlite3.Connection) -> dict[str, Any]:
    """Seasons -> weeks present, the latest week, and the config suffixes."""
    rows = connection.execute(
        """
        SELECT p.season, p.week,
               COUNT(*) AS n_games,
               COUNT(o.game_id) AS n_settled,
               MAX(p.generated_at) AS generated_at
        FROM predictions p
        LEFT JOIN outcomes o USING (game_id)
        GROUP BY p.season, p.week
        ORDER BY p.season, p.week
        """
    ).fetchall()

    seasons: dict[int, dict[str, Any]] = {}
    weeks_flat: list[dict[str, Any]] = []
    for row in rows:
        season = int(row["season"])
        kind = "live" if season == LIVE_SEASON else "backtest"
        seasons.setdefault(season, {"season": season, "kind": kind, "weeks": []})
        seasons[season]["weeks"].append(int(row["week"]))
        weeks_flat.append(
            {
                "season": season,
                "week": int(row["week"]),
                "kind": kind,
                "n_games": int(row["n_games"]),
                "n_settled": int(row["n_settled"]),
                "generated_at": row["generated_at"],
            }
        )

    latest = None
    if weeks_flat:
        # The most recent slate, not the most recently *written* row: a backfill
        # re-logs old weeks last, so ordering by generated_at would land on a
        # 2023 game. (season, week) is the schedule's own order.
        newest = max(weeks_flat, key=lambda w: (w["season"], w["week"]))
        latest = {k: newest[k] for k in ("season", "week", "kind", "n_games", "n_settled")}

    return {
        "seasons": list(seasons.values()),
        "latest": latest,
        "config_suffixes": config_suffixes(connection),
        "live_season": LIVE_SEASON,
    }


def _members(connection: sqlite3.Connection, game_id: str, model_version: str) -> list[dict]:
    # `rowid` order is insertion order, and `record_predictions` writes the
    # members in `MEMBER_ORDER` - so the board and the game page show logreg
    # first and elo last without either end hard-coding that list.
    rows = connection.execute(
        """
        SELECT member, home_win_prob, pick
        FROM member_predictions
        WHERE game_id = ? AND model_version = ?
        ORDER BY rowid
        """,
        (game_id, model_version),
    ).fetchall()
    return [
        {"member": r["member"], "home_win_prob": r["home_win_prob"], "pick": r["pick"]}
        for r in rows
    ]


def _outcome_of(row: sqlite3.Row) -> dict | None:
    if row["home_win"] is None:
        return None
    return {
        "home_score": row["home_score"],
        "away_score": row["away_score"],
        "home_win": row["home_win"],
        "pick_correct": _pick_correct(row["pick"], row["home_team"], row["home_win"]),
    }


def _base_row(connection: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    """The fields the weekly board and the single-game page both carry.

    ``week()`` adds ``generated_at`` and ``has_explanation``; ``game()`` adds
    ``record`` and ``available_versions``.
    """
    return {
        "game_id": row["game_id"],
        "model_version": row["model_version"],
        "config": suffix_of(row["model_version"]),
        "season": int(row["season"]),
        "week": int(row["week"]),
        "gameday": row["gameday"],
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "pick": row["pick"],
        "home_win_prob": row["home_win_prob"],
        "confidence": row["confidence"],
        "agreement": row["agreement"],
        "elo_prob": row["elo_prob"],
        "market_prob": row["market_prob"],
        "members": _members(connection, row["game_id"], row["model_version"]),
        "outcome": _outcome_of(row),
    }


def week(connection: sqlite3.Connection, season: int, week_no: int) -> dict[str, Any] | None:
    """One week's board: every predicted game, its member votes, its outcome.

    When a week was logged under more than one config (a rerun with a changed
    recipe), the most recently generated row for each game wins - the same rule
    the CLI's picks table would show on a rerun.
    """
    game_rows = connection.execute(
        _GAME_SQL + "WHERE p.season = ? AND p.week = ? "
        "ORDER BY p.gameday, p.game_id, p.generated_at DESC",
        (season, week_no),
    ).fetchall()
    if not game_rows:
        return None

    seen: set[str] = set()
    games = []
    n_settled = 0
    for row in game_rows:
        if row["game_id"] in seen:
            continue
        seen.add(row["game_id"])
        base = _base_row(connection, row)
        n_settled += base["outcome"] is not None
        record = json.loads(row["record"])
        games.append(
            {
                **base,
                "generated_at": row["generated_at"],
                "has_explanation": "explanation" in record,
            }
        )

    kind = "live" if season == LIVE_SEASON else "backtest"
    return {
        "ref": {
            "season": season,
            "week": week_no,
            "kind": kind,
            "n_games": len(games),
            "n_settled": n_settled,
        },
        "games": games,
    }


def game(
    connection: sqlite3.Connection, game_id: str, model_version: str | None
) -> dict[str, Any] | None:
    """One game. Newest config by default; ``model_version`` pins an exact run."""
    versions = [
        r["model_version"]
        for r in connection.execute(
            "SELECT model_version FROM predictions WHERE game_id = ? "
            "ORDER BY generated_at DESC",
            (game_id,),
        )
    ]
    if not versions:
        return None
    chosen = model_version or versions[0]
    if chosen not in versions:
        return None

    row = connection.execute(
        _GAME_SQL + "WHERE p.game_id = ? AND p.model_version = ?",
        (game_id, chosen),
    ).fetchone()

    return {
        **_base_row(connection, row),
        "record": json.loads(row["record"]),
        "available_versions": versions,
    }
