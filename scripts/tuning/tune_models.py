"""Hyperparameter grids for the five base estimators, on training seasons only.

The D-11 pattern, extended from Elo to the model stack (D-16). Each estimator
gets a small pre-declared grid, scored by log loss over a five-fold
``TimeSeriesSplit`` **inside 2006-2015**, and the winner is frozen into
:data:`nflpred.config.MODEL_PARAMS` with the grid quoted beside it.

Three rules keep this auditable rather than a moving target:

* **Validation sees no tuning decision.** 2016-2018 (calibration) and 2019-2021
  (validation) are not scored here, and 2022-2025 is not opened. A
  hyperparameter chosen on validation makes every Phase 4 number a selection
  artefact rather than a measurement.
* **Folds are time-ordered.** ``TimeSeriesSplit`` runs over the *unexpanded*
  game rows in kickoff order, and D-4's tie expansion happens inside each fold.
  Expanding first would reorder the rows and hand the splitter a shuffled
  series - the exact thing constraint 2 forbids.
* **The grid is not re-searched at a checkpoint.** Its output is quoted in
  ``src/nflpred/config.py``. This script exists so that comment can be reproduced.

Both feature sets are scored, because the same grid need not win on 16 columns
and on 29. Writes nothing. Run::

    uv run python scripts/tune_models.py
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping, Sequence
from itertools import product
from typing import Any

import numpy as np
import polars as pl
from sklearn.model_selection import TimeSeriesSplit

from nflpred.config import FEATURE_MATRIX_PATH, TRAIN_SEASONS
from nflpred.evaluate import log_loss
from nflpred.features.build import (
    CORE_FEATURES,
    PHASE3_FEATURES,
    to_training_arrays,
)
from nflpred.modeling.base import BUILDERS, fit_weighted, split_frame

pl.Config.set_tbl_rows(30)
pl.Config.set_tbl_width_chars(140)

#: Folds inside the training seasons. Five puts roughly two seasons in each
#: expanding fold's held-out block, which is the smallest unit that is still a
#: meaningful sample of NFL weeks.
N_SPLITS: int = 5

#: The grids. Deliberately small - at most 24 cells each, with the tree count
#: *fixed* and learning rate, depth and leaf size varied. On 2,670 games the
#: number of trees is not the interesting axis; how much each one is allowed to
#: say is. Fixing it also keeps this script minutes rather than hours.
#:
#: These were re-centred **once**, before anything was frozen. The first pass
#: gridded the three boosters over ``learning_rate ∈ {0.01, 0.03, 0.05}`` and
#: depths ``{2, 3, 4}`` / ``{4, 8, 16}`` leaves, and every winner landed on the
#: lowest-capacity corner with loss rising monotonically away from it - a grid
#: reporting that its own optimum is outside itself. The ranges below extend
#: downward to bracket it. Freezing a boundary winner would have been freezing
#: the edge of the search rather than a measured choice.
GRIDS: dict[str, dict[str, tuple[Any, ...]]] = {
    # Regularisation strength across three orders of magnitude. 2,670 rows and
    # 16-29 correlated columns is exactly the regime where C matters.
    "logreg": {"C": (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)},
    # 500 trees fixed; the two knobs that stop a forest memorising a small set.
    "random forest": {
        "n_estimators": (500,),
        "max_depth": (3, 4, 6, 8, None),
        "min_samples_leaf": (5, 10, 20, 40),
    },
    "xgboost": {
        "n_estimators": (400,),
        "learning_rate": (0.003, 0.01, 0.03),
        "max_depth": (1, 2, 3),
        "subsample": (0.8, 1.0),
        "colsample_bytree": (0.8,),
    },
    "lightgbm": {
        "n_estimators": (400,),
        "learning_rate": (0.003, 0.01, 0.03),
        "num_leaves": (2, 4, 8),
        "min_child_samples": (20, 40),
    },
    "catboost": {
        "iterations": (400,),
        "learning_rate": (0.003, 0.01, 0.03),
        "depth": (1, 2, 4),
        "l2_leaf_reg": (3.0, 10.0),
    },
}

FEATURE_SETS: dict[str, Sequence[str]] = {
    "core (16)": CORE_FEATURES,
    "phase3 (29)": PHASE3_FEATURES,
}


def _cells(grid: Mapping[str, tuple[Any, ...]]) -> Iterator[dict[str, Any]]:
    """Every combination in ``grid``, as keyword dicts."""
    keys = list(grid)
    for values in product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values, strict=True))


def _cv_log_loss(
    name: str, params: Mapping[str, Any], train: pl.DataFrame, features: Sequence[str]
) -> float:
    """Mean held-out log loss over the time-ordered folds.

    Each fold refits from scratch. The calibrator is *not* fitted here: this
    grid chooses a base estimator, and 2016-2018 - the split the calibrator
    owns - is not in scope (D-5).
    """
    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    scores = []

    for fit_idx, score_idx in splitter.split(np.arange(train.height)):
        fold_fit = train[fit_idx.tolist()]
        fold_score = train[score_idx.tolist()]

        x, y, w = to_training_arrays(fold_fit, features)
        estimator = fit_weighted(BUILDERS[name](params), x, y, w)

        probabilities = estimator.predict_proba(
            fold_score.select(features).to_numpy().astype(float)
        )[:, 1]
        scores.append(log_loss(fold_score["home_win"].to_numpy(), probabilities))

    return float(np.mean(scores))


def _tune(name: str, train: pl.DataFrame, features: Sequence[str]) -> pl.DataFrame:
    """Score every cell of one estimator's grid on one feature set."""
    rows = []
    for params in _cells(GRIDS[name]):
        started = time.perf_counter()
        rows.append(
            {
                **{k: str(v) for k, v in params.items()},
                "cv_log_loss": _cv_log_loss(name, params, train, features),
                "seconds": time.perf_counter() - started,
            }
        )

    return (
        pl.DataFrame(rows)
        .with_columns(pl.col("cv_log_loss").round(4), pl.col("seconds").round(1))
        .sort("cv_log_loss")
    )


def main() -> None:
    if not FEATURE_MATRIX_PATH.exists():
        msg = f"{FEATURE_MATRIX_PATH.name} not built. Run `python -m nflpred.features.build`."
        raise FileNotFoundError(msg)

    train = split_frame(pl.read_parquet(FEATURE_MATRIX_PATH), "train")

    print(
        f"Hyperparameter grids - {N_SPLITS}-fold TimeSeriesSplit inside "
        f"{TRAIN_SEASONS[0]}-{TRAIN_SEASONS[1]} only ({train.height:,} games). "
        f"Calibration, validation and test are not read.\n"
    )

    winners: dict[str, dict[str, str]] = {}

    for set_name, features in FEATURE_SETS.items():
        for name in BUILDERS:
            grid = _tune(name, train, features)
            print(f"--- {name} on {set_name}, {grid.height} cells")
            print(grid)
            best = grid.row(0, named=True)
            winners.setdefault(name, {})[set_name] = ", ".join(
                f"{k}={v}" for k, v in best.items() if k not in {"cv_log_loss", "seconds"}
            ) + f"   -> {best['cv_log_loss']:.4f}"
            print()

    print("=" * 70)
    print("winners, for pasting into src/nflpred/config.py")
    print("=" * 70)
    for name, per_set in winners.items():
        print(f"\n{name}")
        for set_name, line in per_set.items():
            print(f"  {set_name:12s} {line}")

    print(
        "\nWhere the two feature sets disagree, the frozen constant is one per "
        "estimator: a hyperparameter that flips between 16 and 29 columns is "
        "reading noise, not a real preference. Record which, and why."
    )


if __name__ == "__main__":
    main()
