"""Phase 2 checkpoint: the baseline every later phase has to beat.

A scaled, L2-penalised logistic regression on ten features, fit on 2006-2015
and reported on 2019-2021. Calibration seasons (2016-2018) and the test set
(2022-2025) are not touched - the first belongs to Phase 4, the second is
opened once, at the end.

The point is not the model. The point is an honest number produced by the
rolling machinery in :mod:`nflpred.features.rolling`, so that a Phase 3 feature
which does not improve on it can be called out rather than tuned around.

Run::

    uv run python -m nflpred.features.build          # if the matrix is stale
    uv run python scripts/phase2_baseline.py
"""

from __future__ import annotations

import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nflpred.config import FEATURE_MATRIX_PATH, TRAIN_SEASONS, VAL_SEASONS
from nflpred.evaluate import (
    always_home,
    market_probability,
    metrics_table,
    reliability_diagram,
)
from nflpred.features.build import BASELINE_FEATURES, to_training_arrays
from nflpred.modeling.base import split_frame

from _shared import report_path, report_written, require_matrix

pl.Config.set_tbl_rows(20)
pl.Config.set_tbl_width_chars(120)


def _model() -> Pipeline:
    """Scaled inputs and L2, exactly as the spec's base-estimator list says.

    L2 is left implicit: scikit-learn 1.8 deprecated the ``penalty`` keyword and
    made L2 with ``C`` the default, so passing ``penalty="l2"`` now warns while
    doing nothing. ``C=1.0`` is that L2 penalty.

    No hyperparameter search. A tuned baseline is a worse baseline: the bar is
    supposed to be what a plain model gets from these ten features.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=1000)),
        ]
    )


def main() -> None:
    require_matrix(FEATURE_MATRIX_PATH)
    matrix = pl.read_parquet(FEATURE_MATRIX_PATH)
    train, val = split_frame(matrix, "train"), split_frame(matrix, "val")

    print(
        f"train {TRAIN_SEASONS[0]}-{TRAIN_SEASONS[1]}: {train.height:,} games   "
        f"validation {VAL_SEASONS[0]}-{VAL_SEASONS[1]}: {val.height:,} games   "
        f"(calibration and test untouched)"
    )

    x_train, y_train, w_train = to_training_arrays(train)
    model = _model()
    model.fit(x_train, y_train, lr__sample_weight=w_train)

    # ------------------------------------------------------------ inspection
    coefficients = pl.DataFrame(
        {
            "feature": list(BASELINE_FEATURES),
            "coefficient": model.named_steps["lr"].coef_[0],
            "train_sd": model.named_steps["scale"].scale_,
        }
    ).with_columns(
        pl.col("coefficient").round(4),
        pl.col("train_sd").round(4),
        # Coefficients are on the standardised scale, so magnitude is directly
        # comparable across features - that is the whole reason for the scaler.
        abs_coefficient=pl.col("coefficient").abs(),
    ).sort("abs_coefficient", descending=True).drop("abs_coefficient")

    print(f"\nintercept (home-field advantage): {model.named_steps['lr'].intercept_[0]:.4f}")
    print(coefficients)

    # -------------------------------------------------------------- scoring
    y_val = val["home_win"].to_numpy()
    p_model = model.predict_proba(val.select(BASELINE_FEATURES).to_numpy().astype(float))[:, 1]
    p_home = always_home(val.height, train["home_win"].mean())
    p_market = market_probability(
        val["home_moneyline"].to_numpy(), val["away_moneyline"].to_numpy()
    )

    entries = [
        ("logistic (10 features)", y_val, p_model),
        ("always pick home", y_val, p_home),
        ("market (de-vigged)", y_val, p_market),
    ]

    print(f"\nvalidation {VAL_SEASONS[0]}-{VAL_SEASONS[1]}")
    print(metrics_table(entries))

    path = reliability_diagram(
        entries,
        report_path("phase2_baseline_reliability.png"),
        title=f"Phase 2 baseline - validation {VAL_SEASONS[0]}-{VAL_SEASONS[1]}",
    )
    report_written(path)


if __name__ == "__main__":
    main()
