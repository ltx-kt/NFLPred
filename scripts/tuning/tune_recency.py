"""Recency half-life grid, scored by walk-forward over the validation seasons.

The D-11 / D-16 pattern, applied to the one hyperparameter Phase 7 introduces.
The spec suggests trying 20-40 weeks and calls recency weighting "likely the
highest-value item" in the in-season list; D-18 is the specific reason to want
it — the calibrator was fitted to a 58.3%-home-win era and scored in a 51.3%
one, and the spec named this as the fix rather than tuning against validation.

Three rules keep it auditable rather than a moving target:

* **The test seasons are not opened.** The grid walks 2019-2021, every fit made
  on games completed before its target week, so no cell sees 2022 or later.
* **The frozen-split numbers see no tuning decision either.** This is a
  walk-forward score end to end; nothing here changes any Phase 2-6 reading.
* **The grid is not re-searched.** Its output is quoted in a comment beside the
  constant it produced. This script exists so that comment can be reproduced,
  not so the constant can drift.

``None`` is the unweighted control and is a real cell of the grid, not a
footnote. Effective sample size is reported next to every row, because that is
the cost side: a half-life short enough to track a shifting home-field advantage
is short enough to throw most of the sample away. **If the control wins, the
finding is that recency weighting does not help here and the constant stays
``None``** — the same rule Phase 5 applied to the ensemble.

Writes nothing. Run::

    uv run python scripts/tune_recency.py
"""

from __future__ import annotations

import time

import polars as pl

from nflpred.backtest import (
    CLOCK_COLUMN,
    effective_sample_size,
    recency_weights,
    walk_forward,
)
from nflpred.config import (
    CALIB_WINDOW_SEASONS,
    FEATURE_MATRIX_GT_PATH,
    RECENCY_HALF_LIFE,
    VAL_SEASONS,
)
from nflpred.evaluate import accuracy, brier, log_loss
from nflpred.modeling.ensemble import ENSEMBLE_FEATURES

#: The grid. The spec's 20-40 bracketed on both sides, plus the control. In
#: league weeks (`nflpred.features.build.league_week_index`), so 20 is roughly one
#: season of football, 60 roughly three and 100 roughly five.
#:
#: 80 and 100 were added after a first pass, before anything was frozen, for the
#: reason D-16 records for the model grids: 60 won that pass with 40 on one side
#: and the unweighted control on the other, and a winner at the edge of its
#: weighted range is a grid reporting that its optimum may lie outside itself.
#: With these two the winner is interior on both sides in half-life, not merely
#: against the ``None`` endpoint.
HALF_LIFE_GRID: tuple[float | None, ...] = (
    10.0,
    15.0,
    20.0,
    30.0,
    40.0,
    60.0,
    80.0,
    100.0,
    None,
)

#: How many target weeks one fit serves. Production and tuning must agree, so
#: this is 1 here because `walk_forward`'s default is 1 — a half-life gridded at
#: a coarser cadence would have been chosen for a different system.
CADENCE: int = 1

pl.Config.set_tbl_rows(30)
pl.Config.set_tbl_width_chars(160)


def _row(matrix: pl.DataFrame, half_life: float | None) -> dict[str, object]:
    started = time.perf_counter()
    result = walk_forward(
        matrix,
        VAL_SEASONS,
        ENSEMBLE_FEATURES,
        half_life=half_life,
        calib_seasons=CALIB_WINDOW_SEASONS,
        cadence=CADENCE,
    )

    y = result.y
    probability = result.probability()

    # Effective sample size at the *last* week of the walk, which is the widest
    # window any fit in it used — the honest "how many games is this worth".
    last = result.weeks.row(-1, named=True)
    completed = matrix.filter(pl.col(CLOCK_COLUMN) < last["league_week"])
    weights = recency_weights(completed, last["league_week"], half_life).to_numpy()

    return {
        "half_life": "none" if half_life is None else f"{half_life:g}",
        "n": len(y),
        "log_loss": log_loss(y, probability),
        "brier": brier(y, probability),
        "accuracy": accuracy(y, probability),
        "eff_n": round(effective_sample_size(weights)),
        "of_n": completed.height,
        "seconds": round(time.perf_counter() - started, 1),
    }


def main() -> None:
    if not FEATURE_MATRIX_GT_PATH.exists():
        msg = (
            f"{FEATURE_MATRIX_GT_PATH.name} not built. Run "
            f"`python -m nflpred.features.build --garbage-time`."
        )
        raise FileNotFoundError(msg)

    matrix = pl.read_parquet(FEATURE_MATRIX_GT_PATH)
    print(
        f"Recency grid — walk-forward across validation "
        f"{VAL_SEASONS[0]}-{VAL_SEASONS[1]} only, refit every week, "
        f"calibration window {CALIB_WINDOW_SEASONS} seasons.\n"
        f"Half-lives are league weeks; the offseason is not counted.\n"
    )

    rows = []
    for half_life in HALF_LIFE_GRID:
        row = _row(matrix, half_life)
        rows.append(row)
        print(
            f"  half_life={row['half_life']:>4}  log loss {row['log_loss']:.4f}  "
            f"({row['seconds']}s)"
        )

    grid = pl.DataFrame(rows).with_columns(
        pl.col("log_loss").round(4),
        pl.col("brier").round(4),
        pl.col("accuracy").round(4),
    ).sort("log_loss")

    print(f"\n{grid}")

    best = grid.row(0, named=True)
    control = next(r for r in rows if r["half_life"] == "none")
    frozen = "none" if RECENCY_HALF_LIFE is None else f"{RECENCY_HALF_LIFE:g}"

    print(
        f"\nbest: half_life={best['half_life']}  -> log loss {best['log_loss']:.4f}   "
        f"unweighted control: {control['log_loss']:.4f}   "
        f"delta {best['log_loss'] - control['log_loss']:+.4f}"
    )
    print(f"frozen in src/nflpred/config.py: RECENCY_HALF_LIFE={frozen}")

    if best["half_life"] != frozen:
        print(
            "\nThe frozen constant no longer matches this grid's winner. Update it "
            "deliberately, or leave it and record why — do not let them drift."
        )


if __name__ == "__main__":
    main()
