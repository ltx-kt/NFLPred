"""Paths, season ranges, split boundaries, and column selections.

Single source of truth for anything referenced by more than one module. Split
boundaries follow Decision log D-1 and D-2 in ``docs/PROJECT_SPEC.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# --------------------------------------------------------------------- paths

#: Repo root. This file is ``<root>/src/nflpred/config.py``, so three levels up.
#: ``data/``, ``artifacts/`` and the rest are siblings of ``src/``, not package
#: resources, so they are located relative to the checkout rather than the
#: installed package.
ROOT: Final[Path] = Path(__file__).resolve().parents[2]

DATA_RAW: Final[Path] = ROOT / "data" / "raw"
DATA_PROCESSED: Final[Path] = ROOT / "data" / "processed"

#: Generated outputs, never inputs: fitted models and the reliability PNGs the
#: phase scripts emit. One gitignored tree (see ``.gitignore``) so a checkout
#: carries source and data only.
ARTIFACTS_DIR: Final[Path] = ROOT / "artifacts"
REPORTS_DIR: Final[Path] = ARTIFACTS_DIR / "reports"

#: Fitted model artifacts. Outputs, not a cache — see D-17. Nothing reads from
#: here to avoid refitting; a phase script always refits and overwrites.
MODELS_DIR: Final[Path] = ARTIFACTS_DIR / "models"

PBP_DIR: Final[Path] = DATA_RAW / "pbp"
TEAM_STATS_DIR: Final[Path] = DATA_RAW / "team_stats"
SCHEDULES_PATH: Final[Path] = DATA_RAW / "schedules.parquet"
MANIFEST_PATH: Final[Path] = DATA_RAW / "manifest.json"

#: The Phase 7 prediction log (D-28). SQLite, in the stdlib, under the already
#: gitignored ``data/`` tree — a weekly pipeline that needs a database server to
#: record what it predicted is a weekly pipeline that will stop being run.
PREDICTIONS_DB: Final[Path] = ROOT / "data" / "predictions.sqlite"

TEAM_GAME_PATH: Final[Path] = DATA_PROCESSED / "team_game.parquet"
FEATURE_MATRIX_PATH: Final[Path] = DATA_PROCESSED / "game_features.parquet"

#: The Vite build output the API serves at ``/`` when it exists (see
#: ``src/nflpred/api/main.py``). Absent in a dev checkout that has not run
#: ``npm run build``; the API simply skips the static mount then.
FRONTEND_DIST: Final[Path] = ROOT / "frontend" / "dist"

#: The garbage-time-filtered twins of the two tables above. Built alongside the
#: unfiltered ones rather than instead of them, so D-3's "compare on validation,
#: keep whichever wins" is a measurement rather than an assertion.
TEAM_GAME_GT_PATH: Final[Path] = DATA_PROCESSED / "team_game_gt.parquet"
FEATURE_MATRIX_GT_PATH: Final[Path] = DATA_PROCESSED / "game_features_gt.parquet"

# ------------------------------------------------------------------- seasons

#: First season with nflverse play-by-play.
FIRST_SEASON: Final[int] = 1999

#: Most recent season with completed results. Everything at or below this is
#: immutable and never re-fetched once cached.
LAST_COMPLETED_SEASON: Final[int] = 2025

#: The season we actually predict. Its schedule exists; its results do not.
LIVE_SEASON: Final[int] = 2026

#: Seasons pulled and cached. Wider than the training matrix on purpose: the
#: pre-2006 seasons cost little and give Elo a burn-in runway (D-2).
#:
#: Extended through :data:`LIVE_SEASON` at Phase 7 — the live path needs the
#: season it is predicting to be pullable, and its play-by-play is legitimately
#: absent until week 1 kicks off. `nflpred.ingest.read_pbp` skips a live season with
#: no plays yet rather than raising, which is the one place that absence is
#: expected rather than a broken cache.
INGEST_SEASONS: Final[tuple[int, ...]] = tuple(range(FIRST_SEASON, LIVE_SEASON + 1))

#: First season admitted to the feature matrix. `cpoe` does not exist before
#: 2006, and 1999-2001 was a 31-team league with different divisions (D-2).
MATRIX_START_SEASON: Final[int] = 2006

# -------------------------------------------------------------------- splits

#: Inclusive (start, end) season bounds. Time-ordered and non-overlapping —
#: constraint 2 forbids any shuffled split. See D-1 for why test runs to 2025,
#: and D-5 for why calibration owns 2016-2018 rather than sharing validation.
TRAIN_SEASONS: Final[tuple[int, int]] = (2006, 2015)
CALIB_SEASONS: Final[tuple[int, int]] = (2016, 2018)
VAL_SEASONS: Final[tuple[int, int]] = (2019, 2021)
TEST_SEASONS: Final[tuple[int, int]] = (2022, 2025)

#: How many trailing seasons the walk-forward harness gives its calibrator
#: (D-26). Three, because that is what D-5 gave it on the frozen split — the
#: structure moves through time, it does not change shape. The seasons are held
#: out of the base fit, exactly as 2016-2018 were.
CALIB_WINDOW_SEASONS: Final[int] = 3

# ---------------------------------------------------------- column selection

#: 36 of 372 play-by-play columns. Selected at load time — the full frame is
#: ~50 MB/season on disk and we need roughly a tenth of it.
#:
#: `vegas_wp` / `wp` were deliberately absent at Phase 1 (D-3) and were added at
#: Phase 3 to build the garbage-time variant. That widening changed the column
#: hash and forced a re-pull of all 27 seasons, which is exactly the behaviour
#: the manifest cache exists to produce.
PBP_COLUMNS: Final[tuple[str, ...]] = (
    # keys and time ordering
    "game_id",
    "play_id",
    "season",
    "season_type",
    "week",
    "posteam",
    "defteam",
    "posteam_type",
    "home_team",
    "away_team",
    # play classification — these drive the rate denominators
    "play",
    "special",
    "pass",
    "rush",
    "qb_dropback",
    "qb_kneel",
    "qb_spike",
    "down",
    "ydstogo",
    # core efficiency signal
    "epa",
    "success",
    "yards_gained",
    "first_down",
    # QB quality (cpoe is null before 2006)
    "qb_epa",
    "cpoe",
    "passer_player_id",
    # third down, provided directly rather than derived from down/first_down
    "third_down_converted",
    "third_down_failed",
    # pressure and turnovers
    "sack",
    "interception",
    "fumble_lost",
    # scoring, for reconciliation against schedules
    "posteam_score_post",
    "defteam_score_post",
    "touchdown",
    # win probability, for the Phase 3 garbage-time variant (D-3, D-9).
    # `vegas_wp` needs a spread line and so is sparser in early seasons; `wp`
    # is model-only and is the fallback.
    "vegas_wp",
    "wp",
)

#: 29 of 46 schedule columns. Situational features come free here — the spec is
#: explicit that these should not be recomputed from play-by-play.
SCHEDULE_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "gametime",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "result",
    "overtime",
    "location",
    "home_rest",
    "away_rest",
    "div_game",
    "spread_line",
    "total_line",
    "home_moneyline",
    "away_moneyline",
    "roof",
    "surface",
    "temp",
    "wind",
    "stadium_id",
    "home_qb_id",
    "away_qb_id",
    "home_qb_name",
    "away_qb_name",
)

# ----------------------------------------------------------------- constants

#: A play gaining this many yards or more counts as explosive (Tier 2 feature).
EXPLOSIVE_YARDS: Final[int] = 20

#: Rolling window lengths, in games. Four is "recent form", eight is roughly
#: half a season. Windows deliberately cross season boundaries (D-6), so every
#: matrix row from 2006 on has a full window behind it.
ROLLING_WINDOWS: Final[tuple[int, ...]] = (4, 8)

#: Win-probability band a play must fall inside to count toward the
#: garbage-time-filtered aggregates (D-9). Plays with *null* win probability are
#: kept, not dropped: an unjudgeable play is not a garbage-time play, and
#: dropping them would silently empty whichever seasons lack `vegas_wp`.
WP_BAND: Final[tuple[float, float]] = (0.05, 0.95)

#: Statistics that get a season-to-date window on top of the 4/8-game ones.
#: Restricted on purpose — a season-to-date third-down-attempt *count* is noise,
#: and rolling all 23 statistics would add 46 columns for maybe three signals.
STD_WINDOW_STATS: Final[tuple[str, ...]] = (
    "epa_per_play",
    "success_rate",
    "pass_epa_per_play",
    "rush_epa_per_play",
    "points",
)

# ---------------------------------------------------------------------- elo
#
# Frozen from `scripts/tune_elo.py`, which grids K x HFA x MOV by log loss of
# the Elo-only probability on the **training seasons only** (2006-2015). The
# grid is not re-searched: validation must never see a tuning decision (D-11).
#
# Grid result (2006-2015, 2,670 games, log loss; MOV = margin-of-victory
# multiplier). The 538-style defaults won outright, so these constants are the
# grid's answer rather than an assumption it happened not to contradict.
#
#     MOV on                        MOV off
#     K   HFA=45  55      65        HFA=45  55      65
#     12  0.6288  0.6287  0.6294    0.6476  0.6477  0.6485
#     16  0.6265  0.6263  0.6269    0.6425  0.6425  0.6433
#     20  0.6262 *0.6259* 0.6265    0.6389  0.6389  0.6396
#     24  0.6272  0.6269  0.6275    0.6364  0.6363  0.6371
#
# The MOV multiplier is worth ~0.013 of log loss — an order of magnitude more
# than any K or HFA choice, which move it by ~0.003 across the whole grid. The
# surface is flat enough around the winner that the exact cell barely matters;
# turning MOV off is the only decision here that would cost anything.

#: Starting rating for a team with no history.
ELO_INIT: Final[float] = 1500.0

#: Expansion teams enter below the mean rather than at it — HOU 2002 is the only
#: case inside the ingest range, and a 1500 expansion team would be rated as an
#: average one for most of its first season.
ELO_EXPANSION_INIT: Final[float] = 1300.0

#: Update rate, in rating points per unit of surprise.
ELO_K: Final[float] = 20.0

#: Home-field advantage, in rating points, added to the home team's rating
#: inside the expectation. Applied only at non-neutral sites.
ELO_HFA: Final[float] = 55.0

#: 538's log-margin multiplier, which scales the update by margin of victory
#: while damping the autocorrelation that lets good teams run away with it.
ELO_MOV: Final[bool] = True

#: Fraction of the distance to :data:`ELO_INIT` each rating regresses between
#: seasons: ``1500 + (1 - r) * (elo - 1500)`` with ``r = 1/3``.
ELO_SEASON_REGRESSION: Final[float] = 1.0 / 3.0

#: How much of a team's turnover margin is signal, as a fraction (Tier 3).
#:
#: Frozen from `nflpred.features.build.turnover_reliability`, which split-half
#: correlates each team-season's odd and even games across 2006-2015:
#:
#:     unfiltered team-game table   0.1328
#:     garbage-time filtered        0.0327
#:
#: Roughly a seventh of recent turnover margin carries into the next stretch of
#: games. That is the measured form of the spec's "do not let the model lean on
#: raw recent turnover margin", and `net_turnovers_r8_diff` is scaled by it.
#:
#: Frozen rather than recomputed at build time for two reasons. It is a quantity
#: fitted on the training set, so refitting it on every build makes the feature
#: depend on data that has nothing to do with the game being predicted. And the
#: two matrices must differ in exactly one thing — the win-probability filter —
#: for D-9 to be a controlled comparison, which a per-table factor would break.
TURNOVER_RELIABILITY: Final[float] = 0.1328

#: Playoff round codes in `game_type`. Anything not "REG" is postseason.
POSTSEASON_TYPES: Final[frozenset[str]] = frozenset({"WC", "DIV", "CON", "SB"})

#: Relocated franchises, mapped to their current code.
#:
#: The two sources disagree: `load_schedules` uses the era-correct abbreviation
#: (OAK through 2019, SD through 2016, STL through 2015) while `load_pbp` uses
#: the current one for every season. Left alone this silently drops ~880 games
#: on the join.
#:
#: Canonicalising to the *current* code is the deliberate choice: it keeps each
#: franchise's history as one continuous series, so rolling windows and Elo
#: ratings carry across a relocation instead of resetting. `game_id` keeps the
#: historical abbreviation and is not rewritten — it agrees across both sources
#: already, and it is only a key.
TEAM_ALIASES: Final[dict[str, str]] = {
    "OAK": "LV",   # Raiders -> Las Vegas, 2020
    "SD": "LAC",   # Chargers -> Los Angeles, 2017
    "STL": "LA",   # Rams -> Los Angeles, 2016
}


# ------------------------------------------------------------------- models
#
# Frozen from `scripts/tune_models.py` — the D-11 pattern extended to the model
# stack (D-16). Every grid is scored by log loss over a five-fold
# `TimeSeriesSplit` **inside the training seasons only** (2006-2015). Neither
# the calibration split nor validation sees a hyperparameter decision, so a
# Phase 4 number is a measurement rather than a selection artefact.
#
# Each grid is quoted in the comment above the constants it produced. The
# script is not re-run at a checkpoint: it exists so these comments can be
# reproduced, not so the constants can drift.

#: Seeds every estimator that has one, plus anything else that samples. One
#: constant, so "same inputs, same numbers" is checkable rather than hoped for.
RANDOM_STATE: Final[int] = 1999

#: Thread count for the estimators that parallelise. Fixed rather than -1
#: because the number of threads changes the order floats are accumulated in,
#: and a model whose predictions depend on the machine it was fitted on cannot
#: be verified by a fingerprint (D-17).
#:
#: The random forest is pinned to one thread instead, in `nflpred.modeling.base`, for
#: a stronger reason: fixing sklearn's thread *count* does not fix its
#: accumulation *order*, and the forest is the one estimator here that sums into
#: a shared buffer from workers.
N_JOBS: Final[int] = 4

# What the grids actually showed, before the numbers: **every estimator wants
# less capacity than its defaults give it.** A first pass gridded the three
# boosters over learning rate {0.01, 0.03, 0.05} and depth {2, 3, 4} and put
# every winner on the lowest-capacity corner with loss rising monotonically
# away from it — a grid saying its own optimum lay outside itself. The ranges
# were extended downward once, before anything was frozen, and every winner
# below is now interior to its grid. On 2,670 games that is the whole story:
# XGBoost's best tree is a *stump*, and the forest's best depth is 4.
#
# Grids are scored on 16 columns (`CORE_FEATURES`) and again on 29
# (`PHASE3_FEATURES`), because the same setting need not win on both. They
# mostly did; where they did not, the difference is an override below rather
# than a single compromise constant. Applying 16-column hyperparameters to the
# 29-column model would tilt the feature-count comparison the checkpoint exists
# to settle.

#: Frozen hyperparameters, per estimator. Anything not named here is the
#: library default, except the housekeeping arguments (seeds, thread counts,
#: verbosity) that :mod:`nflpred.modeling.base` sets on every fit.
#:
#: logreg — C, on 16 columns. Interior winner; the surface is nearly flat, with
#: 0.0021 of log loss between the best and worst cell:
#:
#:     C          0.01    0.03    0.1     0.3     1.0     3.0
#:     log loss   0.6250 *0.6240* 0.6245  0.6252  0.6258  0.6261
#:
#: random forest — 500 trees fixed, max_depth x min_samples_leaf, on 16
#: columns. The flattest grid of the five: 0.0031 separates all 20 cells, and
#: the only choice that costs anything is letting the trees grow unbounded at
#: a small leaf size (None/5, 0.6297). Depth 4 wins on both feature sets.
#:
#:              leaf=5   10      20      40
#:     depth 3  0.6269  0.6271  0.6268  0.6270
#:     depth 4 *0.6266* 0.6270  0.6266  0.6270
#:     depth 6  0.6270  0.6275  0.6268  0.6269
#:     depth 8  0.6277  0.6272  0.6274  0.6270
#:     None     0.6297  0.6280  0.6273  0.6269
#:
#: xgboost — 400 trees fixed, learning_rate x max_depth at subsample 0.8 (which
#: beat 1.0 in 8 of 9 pairs), on 16 columns. Depth 1 is the floor of the grid
#: because it is the floor of the model: a depth-1 tree is a stump, and the
#: data preferring stumps over depth-2 is the finding, not a truncation.
#:
#:              lr=0.003  0.01     0.03
#:     depth 1  0.6431   *0.6280*  0.6296
#:     depth 2  0.6356    0.6292   0.6404
#:     depth 3  0.6345    0.6328   0.6534
#:
#: lightgbm — 400 trees fixed, learning_rate x num_leaves at
#: min_child_samples 40, on 16 columns:
#:
#:               lr=0.003  0.01     0.03
#:     leaves 2  0.6449    0.6299   0.6295
#:     leaves 4  0.6369   *0.6293*  0.6441
#:     leaves 8  0.6374    0.6379   0.6681
#:
#: catboost — 400 iterations fixed, learning_rate x depth at l2_leaf_reg 3,
#: on 16 columns:
#:
#:              lr=0.003  0.01     0.03
#:     depth 1  0.6463    0.6288   0.6283
#:     depth 2  0.6400   *0.6271*  0.6308
#:     depth 4  0.6362    0.6283   0.6428
MODEL_PARAMS: Final[dict[str, dict[str, object]]] = {
    "logreg": {"C": 0.03},
    "random forest": {"n_estimators": 500, "max_depth": 4, "min_samples_leaf": 5},
    "xgboost": {
        "n_estimators": 400,
        "learning_rate": 0.01,
        "max_depth": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "lightgbm": {
        "n_estimators": 400,
        "learning_rate": 0.01,
        "num_leaves": 4,
        "min_child_samples": 40,
    },
    "catboost": {"iterations": 400, "learning_rate": 0.01, "depth": 2, "l2_leaf_reg": 3.0},
}

#: Where the 29-column grid disagreed with the 16-column one, keyed by feature
#: count. Only two settings moved, both toward *less* regularisation on the
#: wider set — which is what one would expect if the extra 13 columns are
#: mostly noise the estimator has to average away rather than signal.
#:
#:     logreg      C 0.03 -> 0.01                  (0.6265 vs 0.6280 at 0.03)
#:     lightgbm    min_child_samples 40 -> 20      (0.6332 vs 0.6368 at 40)
#:
#: Random forest, XGBoost and CatBoost picked identical cells on both sets.
#: Feature sets with no entry here — `wide_features()` at 104 — use the frozen
#: constants above untuned, which is a handicap worth stating whenever that row
#: is reported.
MODEL_PARAM_OVERRIDES: Final[dict[int, dict[str, dict[str, object]]]] = {
    29: {
        "logreg": {"C": 0.01},
        "lightgbm": {"min_child_samples": 20},
    },
}


# ------------------------------------------------------------------ recency
#
# Frozen from `scripts/tune_recency.py` — the D-11 / D-16 pattern applied to the
# one hyperparameter Phase 7 introduces. The grid is scored by log loss of the
# headline stack over a **walk-forward** across the validation seasons only
# (2019-2021, every fit on games completed before the target week), so the test
# split sees no tuning decision and neither does any frozen-split number.
#
# Grid result (validation 2019-2021, 821 games, 64 target weeks per cell,
# refit every week). `eff_n` is Kish's effective sample size at the last fit of
# the walk, out of the 4,291 completed games that fit could see — the cost side
# of the trade, since a shorter half-life buys recency by throwing sample away.
#
#     half_life   log loss   brier    accuracy   eff_n / 4,291
#     10          0.6375     0.2229   0.6547       385
#     15          0.6341     0.2213   0.6559       566
#     20          0.6327     0.2207   0.6474       747
#     30          0.6324     0.2205   0.6449     1,110
#     40          0.6323     0.2205   0.6449     1,467
#     60         *0.6318*    0.2203   0.6474     2,120
#     80          0.6321     0.2204   0.6486     2,640
#     100         0.6321     0.2204   0.6474     3,027
#     none        0.6326     0.2206   0.6425     4,291
#
# **The weighting wins, and it wins by 0.0008 of log loss.** That is thin — twice
# the ensemble's own margin over the best single model (D-21), which this project
# already reports as barely worth its complexity — but it is consistent: 60 beats
# the unweighted control on log loss, Brier *and* accuracy, and the surface has
# one shallow minimum rather than the ragged profile of noise.
#
# The winner is interior on both sides. 80 and 100 were added after a first pass
# put 60 at the edge of the weighted range, the same correction D-16 records for
# the model grids, and made before anything was frozen.
#
# Two findings worth more than the winning cell.
#
# **The spec's suggested 20-40 weeks is too short**, under this clock and on this
# sample. Every cell in that band is at or *below* the unweighted control except
# 30 and 40, which clear it by 0.0002-0.0003. The data wants roughly three
# seasons of memory, not one to two.
#
# **Below ~20 the trade turns decisively bad.** At half-life 10 the fit is worth
# 385 games and loses 0.0057 to the control. That is the risk the plan named
# before the run: an effective sample well under 2,670 games, and it is real —
# it just does not bite until the half-life is well short of where the optimum
# sits.

#: Half-life of the exponential sample-weight decay, in **league weeks** (see
#: `nflpred.features.build.league_week_index` — the offseason is not counted, so 60
#: is roughly three seasons of football, not fourteen months). ``None`` turns the
#: weighting off and is the unweighted control, which the grid above beats.
RECENCY_HALF_LIFE: Final[float | None] = 60.0


def season_split(season: int) -> str:
    """Return which split a season belongs to.

    Returns ``"pre"`` for seasons ingested but excluded from the matrix (D-2)
    and ``"live"`` for the season being predicted.
    """
    if season < MATRIX_START_SEASON:
        return "pre"
    if TRAIN_SEASONS[0] <= season <= TRAIN_SEASONS[1]:
        return "train"
    if CALIB_SEASONS[0] <= season <= CALIB_SEASONS[1]:
        return "calib"
    if VAL_SEASONS[0] <= season <= VAL_SEASONS[1]:
        return "val"
    if TEST_SEASONS[0] <= season <= TEST_SEASONS[1]:
        return "test"
    return "live"
