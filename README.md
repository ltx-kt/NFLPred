# NFL Game Winner Prediction

Predicts straight-up NFL winners with a calibrated six-model ensemble, and - as
an equal deliverable - explains every pick: which features drove it, which cut
against it, and what is driving the confidence level. It refits itself every week
as a season progresses, and every probability it reports has been calibrated and
checked against a reliability curve.

**Status:** the seven-phase build is complete. The test split has been spent;
the live target is the 2026 season. A read-only web dashboard over the
prediction log ships alongside the model.

## Headline

On the held-out 2022-2025 seasons, walk-forward refit every week, the system
picks winners **64.8%** of the time at **0.635** log loss. A from-scratch Elo
baseline gets 64.0%; the Las Vegas closing line with the bookmaker's margin
removed gets 67.3%.

Landing below the market is the correct outcome, not a shortfall: beating Vegas
straight-up on public data is not realistic, and a model that appeared to do so
would be leaking information it would not have on game day. The system ships a
hard tripwire at 72% accuracy that raises an error and halts rather than
reporting a suspiciously good number.

## Results

Walk-forward means: for each target week, refit the whole system on everything
completed before that week's first kickoff, predict, score. It is the only number
that describes what the system *does* rather than what one frozen snapshot scored.

| | 2019-2021 (821 games) | | | 2022-2025 (1,139 games) | | |
|---|---|---|---|---|---|---|
| | accuracy | log loss | Brier | accuracy | log loss | Brier |
| **walk-forward, recency-weighted** | **0.6474** | **0.6318** | **0.2203** | **0.6475** | **0.6349** | **0.2222** |
| walk-forward, unweighted | 0.6425 | 0.6326 | 0.2206 | 0.6449 | 0.6359 | 0.2226 |
| frozen-split stack (Phase 5 build) | 0.6279 | 0.6387 | 0.2239 | 0.6255 | 0.6358 | 0.2227 |
| elo only (raw) | 0.6352 | 0.6422 | 0.2246 | 0.6405 | 0.6361 | 0.2229 |
| always pick home | 0.5134 | 0.6984 | 0.2526 | 0.5536 | 0.6877 | 0.2473 |
| market (de-vigged) | 0.6462 | 0.6107 | 0.2116 | 0.6730 | 0.6075 | 0.2102 |

- **Refitting every week is worth about ten times what the recency weighting on
  top of it is worth.** Against the identical construction on a frozen split,
  walking forward buys 0.0069 of log loss on validation; the half-life buys
  another 0.0008.
- **Accuracy survives the era change; calibration does not.** 0.6474 -> 0.6475
  against 0.6318 -> 0.6349 is a model whose ranking generalizes across a regime
  shift better than its probabilities do.
- **Home-field advantage collapsed mid-dataset** - 58.3% of games in 2016-2018,
  51.3% in 2019-2021 - and the de-vigged market line bends the same way on the
  same games, which is the control that proves the effect is real. That finding
  is why the weekly retraining pipeline exists.

The table above is the summary; the frozen-split Phase 2-6 tables, the recency
half-life grid, and the negative results in full are in the local design docs
(see [Documentation](#documentation)).

## The dashboard

A read-only web view over `data/predictions.sqlite`: a FastAPI JSON API
(`src/nflpred/api/`) and a Vite + React frontend (`frontend/`) that collapse into one
process in production.

| Route | Shows |
|---|---|
| `/` | the weekly board: one row per game, per-member votes, Elo and market anchors |
| `/history` | a browsable walk-forward backtest by season and week |
| `/model` | calibration reliability curve and per-member rolling Brier |
| `/game/:id` | the full output-contract record and explanation for one game |
| `/about` | how to read the board |

**The API never predicts.** Every endpoint is a `SELECT` against the log, which
already carries the calibrated probability, the pick, the confidence band, the
six member votes, and the anchors. `python -m nflpred.predict` is the only writer.
Setup is in the [Quickstart](#quickstart) below; the running API serves its own
live endpoint list and schema at `/docs`.

The `Dockerfile` copies `data/predictions.sqlite` into the image, and `data/` is
gitignored - a fresh clone must run `scripts/ops/seed_predlog.py` to build the log
before `docker build` will succeed.

## What keeps it honest

- **Leak-safe by construction.** Exactly one lagging expression exists in the
  codebase, `nflpred.features.rolling.lagged`, which does a `shift(1)` inside each
  team's time-ordered series before any value is joined to a game row. A suite in
  `tests/test_no_leakage.py` rewrites one game's box score, result, and
  quarterback line to absurd values, rebuilds the whole matrix, and demands that
  game's own row come back unchanged - each paired with a guard that later rows
  *do* move.
- **Time-based splits only.** Fixed season boundaries, cross-validation only
  *inside* the training seasons. Train 2006-2015, calibrate 2016-2018, validate
  2019-2021, test 2022-2025 (touched once).
- **Calibrated confidence.** Every base model is wrapped in
  `CalibratedClassifierCV` fitted on a dedicated split, verified against a
  reliability curve. Log loss and Brier are reported alongside accuracy at every
  checkpoint.
- **Negative results published, not tuned away.** The six-model ensemble beats
  its best single model by 0.0004 of log loss and loses 0.0049 of accuracy (the
  members correlate at a mean of 0.959). The garbage-time filter is a measured
  wash. A 29-column feature set lost to the 16-column set for every estimator.
  All three are reported as results.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv sync

# Build the data layer: cache nflverse pulls, aggregate, assemble the matrix
uv run python -m nflpred.ingest
uv run python -m nflpred.features.pbp_agg
uv run python -m nflpred.features.pbp_agg --garbage-time
uv run python -m nflpred.features.build
uv run python -m nflpred.features.build --garbage-time

# Phase checkpoints (each refits from scratch; none touches the test split)
uv run python scripts/phases/phase1_report.py       # shape, nulls, reconciliation
uv run python scripts/phases/phase2_baseline.py     # the 10-feature bar
uv run python scripts/phases/phase3_features.py     # what each feature group is worth
uv run python scripts/phases/phase4_models.py       # five calibrated estimators
uv run python scripts/phases/phase5_ensemble.py     # soft voting and stacking
uv run python scripts/phases/phase6_explanations.py # attribution + hand-checked games
uv run python scripts/phases/phase7_pipeline.py     # walk-forward, the test touch, the log

# A weekly run
uv run python -m nflpred.predict --season 2026 --week 1 --explain
uv run python -m nflpred.predict --season 2025 --week 8 --json week8.json
uv run python -m nflpred.predict --settle    # join completed results onto the log
uv run python -m nflpred.predict --report    # per-member rolling Brier from the log

# The dashboard
uv run python scripts/ops/seed_predlog.py --seasons 2022 2025   # backfill the log
uv run --group api uvicorn nflpred.api.main:app --reload         # API on :8000
cd frontend && npm install && npm run dev                        # frontend on :5173

# Tuning grids (train seasons only) and checks
uv run python -m nflpred.features.elo        # Elo rating sanity dump
uv run python -m nflpred.features.qb         # QB composite sanity dump
uv run python scripts/tuning/tune_elo.py
uv run python scripts/tuning/tune_models.py
uv run python scripts/tuning/tune_recency.py
uv run pytest
uv run ruff check .
```

## Architecture

Data flows one direction, from raw pull to logged prediction to dashboard:

```
  nflverse (nflreadpy)
        |
        v
  nflpred/ingest.py .............. pull + column-select (36 of ~372) + cache
        |
        v
  data/raw/*.parquet ............. play-by-play, schedules, team stats (+ manifest.json)
        |
        v
  nflpred/features/pbp_agg.py .... one row per (game, team): same-game box score
        |
        +--> features/rolling.py .. leak-safe rolling windows (shift(1))
        +--> features/elo.py ...... from-scratch Elo, pre-game ratings
        +--> features/qb.py ....... QB composite + projected starter
        |
        v
  nflpred/features/build.py ...... one row per game, home-team perspective
        |
        v
  data/processed/game_features.parquet
        |
        v
  nflpred/modeling/base.py ....... 5 estimators, each calibrated on its own split
  nflpred/modeling/ensemble.py ... 5 estimators + Elo into a stacked ensemble
        |
        v
  nflpred/explain.py ............. per-member SHAP, confidence decomposition, narrative
        |
        v
  nflpred/backtest.py ........... walk-forward refit + recency weighting
  nflpred/predict.py ........... weekly CLI (incl. synthesizing unplayed games)
        |
        v
  data/predictions.sqlite ....... every prediction, every member vote, every outcome
        |
        v
  nflpred/api/ + frontend/ ...... read-only dashboard: every endpoint is a SELECT
```

## Layout

```
src/nflpred/
  config.py       paths, seasons, splits, column selections, frozen constants
  ingest.py       nflreadpy loaders + content-hash cache manifest
  evaluate.py     metrics, comparators, reliability diagrams, the 72% tripwire
  backtest.py     recency weights, the weekly fit, the walk-forward harness
  predict.py      week_frame (incl. unplayed games) + the weekly CLI
  predlog.py      the SQLite log: runs, predictions, member_predictions, outcomes
  explain.py      FEATURE_GLOSSARY (30 entries), SHAP attribution, narratives
  features/       pbp_agg.py, rolling.py, elo.py, qb.py, build.py
  modeling/       base.py (factory + calibration), ensemble.py, store.py
  api/            FastAPI: main.py, db.py, queries.py, analytics.py, routes/
frontend/         Vite + React + Tailwind dashboard (5 pages, react-query)
scripts/          phases/ (phase1-7 reports), tuning/ (tune_*), ops/ (seed_predlog.py)
tests/            180 tests across 12 files + conftest.py
data/raw/         cached parquet + manifest.json (gitignored)
data/processed/   team_game{,_gt}.parquet, game_features{,_gt}.parquet
data/predictions.sqlite   the prediction log (gitignored)
artifacts/        fitted models/ + reliability reports/ (outputs, gitignored)
Dockerfile        multi-stage: Node builds frontend/, Python serves both from one uvicorn
```

Roughly 7,400 lines of typed Python in `src/`, 3,400 lines of tests, 2,900 lines
of tuning and reporting scripts, and 1,700 lines of frontend.

## Documentation

The design docs, the full evaluation write-up, and the 32-entry decision log
(a project overview, the results analysis, the build spec, data notes, and the
phase-1 / weekly-run / dashboard guides) are kept outside this repository. The
code and this README are what ships.

## Stack

Python 3.13, [uv](https://docs.astral.sh/uv/) for dependencies and task running,
[Polars](https://pola.rs/) for the entire data and feature layer (numpy and
pandas appear only at the scikit-learn boundary),
[nflreadpy](https://github.com/nflverse/nflreadpy) for raw data (`nfl_data_py` is
deprecated - do not use it). scikit-learn 1.9 for logistic regression, random
forest, calibration, and stacking; XGBoost, LightGBM, and CatBoost as three
boosted members; [SHAP](https://github.com/shap/shap) for per-game attribution,
imported lazily. SQLite (stdlib) for the prediction log. FastAPI + uvicorn for
the API; React + Vite + Tailwind for the frontend. pytest and ruff, with type
annotations enforced on `src/`.
