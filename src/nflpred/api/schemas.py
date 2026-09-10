"""Response models. Field names mirror the log schema in `nflpred.predlog.SCHEMA`.

Kept hand-written and next to the routes rather than generated: seven endpoints
over a fixed four-table schema is not enough surface to justify a codegen step,
and the frontend's ``frontend/src/api/types.ts`` is meant to be diffed against this
file in review.
"""

from __future__ import annotations

from pydantic import BaseModel


class Health(BaseModel):
    """`predlog.summary` counts, plus whether any prediction has an outcome."""

    runs: int
    predictions: int
    member_predictions: int
    outcomes: int
    settled: int
    settled_share: float
    model_versions: list[str]


class WeekRef(BaseModel):
    season: int
    week: int
    kind: str  # "live" (LIVE_SEASON) or "backtest" (a replayed completed week)
    n_games: int
    n_settled: int


class SeasonIndex(BaseModel):
    season: int
    kind: str
    weeks: list[int]


class Index(BaseModel):
    """Everything the frontend nav needs in one call."""

    seasons: list[SeasonIndex]
    latest: WeekRef | None
    config_suffixes: list[str]
    live_season: int


class MemberVote(BaseModel):
    member: str
    home_win_prob: float
    pick: str


class Outcome(BaseModel):
    home_score: int | None
    away_score: int | None
    home_win: float  # D-4 float: a tie settles as 0.5
    pick_correct: float | None  # 1.0 hit, 0.0 miss, 0.5 tie; null if pick was a "no read"


class GameBase(BaseModel):
    """The fields the weekly board and the single-game page both carry.

    `queries._base_row` builds exactly these; `GameRow` and `GameDetail` each
    add two more. Mirrored in `frontend/src/api/types.ts` as `GamePrediction`.
    """

    game_id: str
    model_version: str
    config: str
    season: int
    week: int
    gameday: str
    home_team: str
    away_team: str
    pick: str
    home_win_prob: float
    confidence: str
    agreement: str
    elo_prob: float | None
    market_prob: float | None
    members: list[MemberVote]
    outcome: Outcome | None


class GameRow(GameBase):
    """One row of the weekly board."""

    generated_at: str
    has_explanation: bool


class Week(BaseModel):
    ref: WeekRef
    games: list[GameRow]


class GameDetail(GameBase):
    """A single game: the flat row, the full output-contract record, the outcome.

    ``record`` is the JSON `nflpred.modeling.ensemble.predict_records` produced, with
    the Phase 6 ``explanation`` key merged in when the week was run with
    ``--explain``. Passed through untouched rather than re-modelled - the
    contract is owned by `nflpred.modeling.ensemble`, not by this layer.
    """

    record: dict
    available_versions: list[str]


class MetricRow(BaseModel):
    model: str
    n: int
    accuracy: float
    log_loss: float
    brier: float


class SeasonMetric(BaseModel):
    season: int
    week: int
    metrics: list[MetricRow]


class Performance(BaseModel):
    """Ensemble vs comparators over settled games, overall and per season-week."""

    config: str
    n_settled: int
    overall: list[MetricRow]
    by_week: list[SeasonMetric]
    accuracy_ceiling: float  # the 72% leakage tripwire, for the chart annotation
    market_band: list[float]  # ~[0.66, 0.68] straight-up, for the chart annotation


class CalibrationPoint(BaseModel):
    predicted: float
    observed: float
    n: int


class Calibration(BaseModel):
    config: str
    bins: int
    n: int
    points: list[CalibrationPoint]


class MemberBrier(BaseModel):
    member: str
    n: int
    brier: float
    mean_prob: float


class Members(BaseModel):
    config: str
    window: int
    overall: list[MemberBrier]
    trailing: list[MemberBrier]
    note: str
