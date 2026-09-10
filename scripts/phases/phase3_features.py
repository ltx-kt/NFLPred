"""Phase 3 checkpoint: what each feature group is actually worth.

The Phase 2 bar was one number. This is the same number, decomposed - Elo, then
the QB composite, then the rest of Tier 2 - so that a group which contributes
nothing can be identified as contributing nothing rather than disappearing into
an aggregate improvement.

Every row uses the **same estimator** as Phase 2: ``StandardScaler`` plus
``LogisticRegression(C=1.0)``, no tuning, fit on 2006-2015, scored on 2019-2021.
The only thing that varies between rows is the feature set. That is what makes
the differences attributable.

Calibration (2016-2018) and test (2022-2025) are not touched. Constraint 3's
tripwire is executable here: any variant above
:data:`nflpred.evaluate.ACCURACY_CEILING` halts the run instead of printing a result.

Run::

    uv run python -m nflpred.features.build                  # if the matrix is stale
    uv run python -m nflpred.features.build --garbage-time
    uv run python scripts/phase3_features.py
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nflpred.config import (
    FEATURE_MATRIX_GT_PATH,
    FEATURE_MATRIX_PATH,
    TRAIN_SEASONS,
    TURNOVER_RELIABILITY,
    VAL_SEASONS,
)
from nflpred.evaluate import (
    always_home,
    market_probability,
    metrics_table,
    reliability_diagram,
)
from nflpred.features.build import (
    BASELINE_FEATURES,
    ELO_FEATURES,
    PHASE3_FEATURES,
    PHASE3_FEATURES_MARKET,
    PHASE3_FEATURES_NO_TO,
    QB_FEATURES,
    to_training_arrays,
    wide_features,
)
from nflpred.modeling.base import split_frame

from _shared import checked, report_path, report_written, require_matrix

pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(160)
pl.Config.set_fmt_str_lengths(40)


def _model() -> Pipeline:
    """The Phase 2 estimator, unchanged and untuned.

    Deliberately not improved. If Phase 3's features are worth something, they
    are worth something to the same model that measured the bar; swapping in a
    better estimator at the same time would make the comparison meaningless.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=1000)),
        ]
    )


def _fit_predict(
    train: pl.DataFrame, evaluate: pl.DataFrame, features: Sequence[str]
) -> tuple[np.ndarray, Pipeline]:
    x_train, y_train, w_train = to_training_arrays(train, features)
    model = _model()
    model.fit(x_train, y_train, lr__sample_weight=w_train)

    x_eval = evaluate.select(features).to_numpy().astype(float)
    return model.predict_proba(x_eval)[:, 1], model


def _coefficients(model: Pipeline, features: Sequence[str]) -> pl.DataFrame:
    """Standardised coefficients, largest first.

    Magnitudes are comparable across features because the scaler put them on the
    same footing - which is the whole reason it is in the pipeline.
    """
    return (
        pl.DataFrame(
            {
                "feature": list(features),
                "coefficient": model.named_steps["lr"].coef_[0],
            }
        )
        .with_columns(
            pl.col("coefficient").round(4), magnitude=pl.col("coefficient").abs()
        )
        .sort("magnitude", descending=True)
        .drop("magnitude")
    )


def main() -> None:
    require_matrix(FEATURE_MATRIX_PATH, FEATURE_MATRIX_GT_PATH)

    matrix = pl.read_parquet(FEATURE_MATRIX_PATH)
    matrix_gt = pl.read_parquet(FEATURE_MATRIX_GT_PATH)

    train, val = split_frame(matrix, "train"), split_frame(matrix, "val")
    train_gt, val_gt = split_frame(matrix_gt, "train"), split_frame(matrix_gt, "val")

    print(
        f"train {TRAIN_SEASONS[0]}-{TRAIN_SEASONS[1]}: {train.height:,} games   "
        f"validation {VAL_SEASONS[0]}-{VAL_SEASONS[1]}: {val.height:,} games   "
        f"(calibration and test untouched)"
    )
    print(
        f"turnover shrinkage factor: {TURNOVER_RELIABILITY:.4f} "
        f"(split-half reliability of turnover margin on {TRAIN_SEASONS[0]}-{TRAIN_SEASONS[1]})"
    )

    y_val = val["home_win"].to_numpy()

    # ------------------------------------------------------------- the rungs
    baseline_elo = (*BASELINE_FEATURES, *ELO_FEATURES)
    baseline_elo_qb = (*baseline_elo, *QB_FEATURES)

    rungs: list[tuple[str, Sequence[str], pl.DataFrame, pl.DataFrame]] = [
        ("baseline 10 (Phase 2 bar)", BASELINE_FEATURES, train, val),
        ("  + Elo", baseline_elo, train, val),
        ("  + QB composite", baseline_elo_qb, train, val),
        ("  + Tier 2 (= PHASE3_FEATURES)", PHASE3_FEATURES, train, val),
        ("wide (all legal columns)", wide_features(), train, val),
        ("PHASE3 - turnovers", PHASE3_FEATURES_NO_TO, train, val),
        ("PHASE3 + market (Tier 3)", PHASE3_FEATURES_MARKET, train, val),
        ("PHASE3, garbage-time filtered", PHASE3_FEATURES, train_gt, val_gt),
    ]

    entries: list[tuple[str, np.ndarray, np.ndarray]] = []
    in_sample: list[tuple[str, np.ndarray, np.ndarray]] = []
    models: dict[str, tuple[Pipeline, Sequence[str]]] = {}

    for name, features, fit_on, score_on in rungs:
        label = f"{name} [{len(features)}]"
        probabilities, model = _fit_predict(fit_on, score_on, features)
        entries.append((label, score_on["home_win"].to_numpy(), probabilities))
        models[name] = (model, features)

        # The same model scored on the data it was fit to. Not a result - a
        # diagnostic. A feature set whose training loss falls while its
        # validation loss rises is overfitting, and that is worth being able to
        # see rather than infer.
        fitted = model.predict_proba(fit_on.select(features).to_numpy().astype(float))[:, 1]
        in_sample.append((label, fit_on["home_win"].to_numpy(), fitted))

    # -------------------------------------------------------- comparators
    entries.append(("elo only", y_val, val["elo_prob"].to_numpy()))
    entries.append(("always pick home", y_val, always_home(val.height, train["home_win"].mean())))
    entries.append(
        (
            "market (de-vigged)",
            y_val,
            market_probability(
                val["home_moneyline"].to_numpy(), val["away_moneyline"].to_numpy()
            ),
        )
    )

    # ---------------------------------------------------------- the tripwire
    table = checked(metrics_table(entries), entries)

    print(f"\nvalidation {VAL_SEASONS[0]}-{VAL_SEASONS[1]}   [n features in brackets]")
    print(
        table.join(
            metrics_table(in_sample).select(
                "model", train_log_loss=pl.col("log_loss")
            ),
            on="model",
            how="left",
        )
    )

    # ------------------------------------------------------------ inspection
    model, features = models["  + Tier 2 (= PHASE3_FEATURES)"]
    print(f"\nPHASE3_FEATURES coefficients (intercept {model.named_steps['lr'].intercept_[0]:.4f})")
    print(_coefficients(model, features))

    # Four curves, not eleven: the bar, the headline, Elo alone, and the market.
    plotted = ("baseline 10", "  + Tier 2", "elo only", "market")
    shown = [e for e in entries if e[0].startswith(plotted)]
    path = reliability_diagram(
        shown,
        report_path("phase3_features_reliability.png"),
        title=f"Phase 3 features - validation {VAL_SEASONS[0]}-{VAL_SEASONS[1]}",
    )
    report_written(path)


if __name__ == "__main__":
    main()
