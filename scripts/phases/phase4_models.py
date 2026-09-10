"""Phase 4 checkpoint: what each base estimator is actually worth.

Five estimators, each on two feature sets, each fitted on 2006-2015 and
calibrated on 2016-2018 (D-5, D-13), all scored on 2019-2021 against the same
comparators. This is a **measurement** phase: the deliverable is an honest table
saying which estimators earn their place, not a tuned winner.

Hyperparameters are frozen constants (D-16) chosen by `scripts/tune_models.py`
on the training seasons alone. Nothing here searches anything — a checkpoint
that tunes is a checkpoint that reports a selection artefact.

Three things the spec carried into Phase 4 are discharged by the follow-up rows:
the D-4 tie encoding (which `tests/test_models.py` pins), the D-9 garbage-time
filter re-checked against a stronger estimator, and the feature-count question
Phase 3 raised. The primary matrix is the **filtered** one, which is what D-9
instructed Phase 4 to fit on; the unfiltered follow-up is the re-check.

Test (2022-2025) is not read. Any model above the accuracy ceiling halts the
run rather than reporting (constraint 3).

Run::

    uv run python scripts/phase4_models.py
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
import polars as pl
from sklearn.calibration import CalibratedClassifierCV

from nflpred.config import (
    CALIB_SEASONS,
    FEATURE_MATRIX_GT_PATH,
    FEATURE_MATRIX_PATH,
    MODELS_DIR,
    TRAIN_SEASONS,
    VAL_SEASONS,
)
from nflpred.evaluate import (
    always_home,
    market_probability,
    metrics_table,
    reliability_diagram,
)
from nflpred.features.build import (
    CORE_FEATURES,
    PHASE3_FEATURES,
    wide_features,
)
from nflpred.modeling.base import (
    build_models,
    elo_platt_probability,
    elo_probability,
    fit_all,
    fit_calibrated,
    fit_elo_platt,
    home_win_probability,
    inner_estimator,
    split_frame,
)
from nflpred.modeling.store import load_models, save_models

from _shared import checked, report_path, report_written, require_matrix

pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(160)
pl.Config.set_fmt_str_lengths(44)

#: The two feature sets every estimator is measured on, and the artifact key
#: each one's models are written under.
FEATURE_SETS: dict[str, tuple[str, Sequence[str]]] = {
    "core (16)": ("core16", CORE_FEATURES),
    "phase3 (29)": ("phase3_29", PHASE3_FEATURES),
}

#: The three gradient boosters, for "best booster" — the wide-feature control
#: and the D-9 re-check are both specified against the strongest of them.
BOOSTERS: tuple[str, ...] = ("xgboost", "lightgbm", "catboost")

#: The Phase 3 headline, for the one comparison that matters: scaled logistic
#: regression at C=1.0 on 16 features, uncalibrated, on the unfiltered matrix.
PHASE3_BAR: dict[str, float] = {"log_loss": 0.6377, "accuracy": 0.6352}

#: Rows named here are entries, not models: a `(label, y, probabilities)` triple.
Entry = tuple[str, np.ndarray, np.ndarray]


def _calibrated_elo(calib: pl.DataFrame, evaluate: pl.DataFrame) -> np.ndarray:
    """Elo, Platt-scaled on the calibration seasons.

    Two lines because the scaler itself now lives in `nflpred.modeling.base` — Phase 5
    holds it as an ensemble member, so it is shared rather than copied.
    """
    return elo_platt_probability(fit_elo_platt(calib), elo_probability(evaluate))


def _score(
    model: CalibratedClassifierCV,
    label: str,
    train: pl.DataFrame,
    val: pl.DataFrame,
    features: Sequence[str],
) -> tuple[Entry, Entry]:
    """One model's validation entry and its in-sample twin.

    The in-sample number is a diagnostic, not a result. On 2,670 rows the gap
    between training and validation loss is the main thing worth watching: a
    model whose training loss falls while its validation loss rises is
    memorising the 2006-2015 seasons.
    """
    return (
        (label, val["home_win"].to_numpy(), home_win_probability(model, val, features)),
        (label, train["home_win"].to_numpy(), home_win_probability(model, train, features)),
    )


def _table(
    entries: Sequence[Entry], in_sample: Sequence[Entry], counts: Sequence[str]
) -> pl.DataFrame:
    """Metrics for every entry, plus feature count and in-sample loss.

    ``in_sample`` covers only the leading model rows — comparators have no
    training loss — so it is padded rather than joined by name. Two rows share
    the name ``logreg``, one per feature set, and a join would multiply them.
    """
    table = metrics_table(entries).with_columns(pl.Series("n_feat", counts))
    columns = ["model", "n_feat", "n", "accuracy", "log_loss", "brier"]

    if in_sample:
        train_ll = list(metrics_table(in_sample)["log_loss"])
        padded = train_ll + [None] * (table.height - len(train_ll))
        table = table.with_columns(pl.Series("train_ll", padded, dtype=pl.Float64))
        columns.append("train_ll")

    return table.select(columns)


def main() -> None:
    require_matrix(FEATURE_MATRIX_PATH, FEATURE_MATRIX_GT_PATH)

    # D-9 instructed Phase 4 to fit on the filtered matrix. The unfiltered one
    # is loaded too, as the re-check that decision asked for.
    matrix = pl.read_parquet(FEATURE_MATRIX_GT_PATH)
    matrix_unfiltered = pl.read_parquet(FEATURE_MATRIX_PATH)

    train, calib, val = (split_frame(matrix, s) for s in ("train", "calib", "val"))
    y_val = val["home_win"].to_numpy()

    print(
        f"train {TRAIN_SEASONS[0]}-{TRAIN_SEASONS[1]}: {train.height:,}   "
        f"calibration {CALIB_SEASONS[0]}-{CALIB_SEASONS[1]}: {calib.height:,}   "
        f"validation {VAL_SEASONS[0]}-{VAL_SEASONS[1]}: {val.height:,}   (test untouched)"
    )
    print(f"matrix: {FEATURE_MATRIX_GT_PATH.name} — garbage-time filtered, per D-9\n")

    entries: list[Entry] = []
    in_sample: list[Entry] = []
    counts: list[str] = []
    fitted: dict[str, dict[str, CalibratedClassifierCV]] = {}
    #: Which (estimator, feature set) each of the first rows belongs to. Kept
    #: explicitly rather than reconstructed from row order, because "the best
    #: model" is picked off this and a positional guess would be a silent bug.
    model_rows: list[tuple[str, str]] = []

    # ------------------------------------------------- five estimators x two sets
    started = time.perf_counter()
    for set_name, (key, features) in FEATURE_SETS.items():
        models = fit_all(matrix, features)
        fitted[set_name] = models

        for name, model in models.items():
            validation_entry, train_entry = _score(model, name, train, val, features)
            entries.append(validation_entry)
            in_sample.append(train_entry)
            counts.append(str(len(features)))
            model_rows.append((name, set_name))

        directory = save_models(
            models,
            key,
            matrix,
            features,
            meta={"phase": 4, "feature_set": set_name, "calibration": "sigmoid"},
            matrix_path=FEATURE_MATRIX_GT_PATH,
        )
        # Artifacts are outputs, not a cache (D-17) — so the run that writes them
        # is the run that proves they read back as the same models.
        load_models(key, matrix)
        print(
            f"  wrote {len(models)} models -> "
            f"{directory.relative_to(MODELS_DIR.parent)}  (fingerprint verified)"
        )

    elapsed = time.perf_counter() - started

    # ---------------------------------------------------------- comparators
    for label, probabilities, n_feat in (
        ("elo only (raw)", elo_probability(val), "1"),
        ("elo only (calibrated)", _calibrated_elo(calib, val), "1"),
        ("always pick home", always_home(val.height, train["home_win"].mean()), "-"),
        (
            "market (de-vigged)",
            market_probability(val["home_moneyline"].to_numpy(), val["away_moneyline"].to_numpy()),
            "-",
        ),
    ):
        entries.append((label, y_val, probabilities))
        counts.append(n_feat)

    table = checked(_table(entries, in_sample, counts), entries)

    print(
        f"validation {VAL_SEASONS[0]}-{VAL_SEASONS[1]}   "
        f"({elapsed:.1f}s to fit and calibrate all ten)"
    )
    print(table)
    print(
        f"\nPhase 3 bar (16 features, uncalibrated logreg C=1.0, unfiltered matrix): "
        f"log loss {PHASE3_BAR['log_loss']:.4f}, accuracy {PHASE3_BAR['accuracy']:.4f}"
    )
    _home_field_era(matrix)

    # ------------------------------------------------------------- follow-ups
    losses = table["log_loss"].to_list()
    best_model_name, best_set = _best(model_rows, losses)
    booster_label, booster_set = _best(model_rows, losses, among=BOOSTERS)
    print(
        f"\nbest overall: {best_model_name} on {best_set}   |   "
        f"best booster: {booster_label} on {booster_set}"
    )

    follow_entries: list[Entry] = []
    follow_counts: list[str] = []
    wide = wide_features()

    # 1. Feature-count control: does the best booster want 104 columns?
    #    Its hyperparameters were never gridded at that width (nothing was), so
    #    this row is a control with a handicap, and is read as one.
    wide_model = fit_calibrated(
        build_models(wide)[booster_label], train, calib, wide
    )
    follow_entries.append(
        (f"{booster_label}, wide", y_val, home_win_probability(wide_model, val, wide))
    )
    follow_counts.append(str(len(wide)))

    # 2. D-9 re-check: the same estimator and feature set on the unfiltered
    #    matrix. Everything else is held constant, so the difference is the
    #    win-probability filter and nothing else.
    unfiltered_features = FEATURE_SETS[booster_set][1]
    train_u, calib_u, val_u = (
        split_frame(matrix_unfiltered, s) for s in ("train", "calib", "val")
    )
    unfiltered_model = fit_calibrated(
        build_models(unfiltered_features)[booster_label], train_u, calib_u, unfiltered_features
    )
    follow_entries.append(
        (
            f"{booster_label}, unfiltered matrix",
            val_u["home_win"].to_numpy(),
            home_win_probability(unfiltered_model, val_u, unfiltered_features),
        )
    )
    follow_counts.append(str(len(unfiltered_features)))

    # 3. Calibration diagnostic: isotonic instead of sigmoid, for the best model.
    best_features = FEATURE_SETS[best_set][1]
    isotonic = fit_calibrated(
        build_models(best_features)[best_model_name], train, calib, best_features, method="isotonic"
    )
    follow_entries.append(
        (
            f"{best_model_name}, isotonic",
            y_val,
            home_win_probability(isotonic, val, best_features),
        )
    )
    follow_counts.append(str(len(best_features)))

    # 4. What the calibrator bought: the same fitted base, scored raw. D-5 costs
    #    801 training games, so what it buys should be visible rather than assumed.
    uncalibrated = inner_estimator(fitted[best_set][best_model_name])
    follow_entries.append(
        (
            f"{best_model_name}, uncalibrated base",
            y_val,
            uncalibrated.predict_proba(val.select(best_features).to_numpy().astype(float))[:, 1],
        )
    )
    follow_counts.append(str(len(best_features)))

    follow_table = checked(_table(follow_entries, [], follow_counts), follow_entries)

    print("\nfollow-ups")
    print(follow_table)

    # ------------------------------------------------------------- diagram
    #
    # Four curves: the best calibrated model, that same model's uncalibrated
    # base, Elo raw, and the market. The pair is the point — if D-5's calibrator
    # is doing its job the calibrated curve sits closer to the diagonal than the
    # raw one, and if it does not, the calibrator is misfitted rather than
    # merely unhelpful.
    best_index = model_rows.index((best_model_name, best_set))
    plotted: list[Entry] = [
        (f"{best_model_name} ({best_set}), calibrated", *entries[best_index][1:]),
        follow_entries[-1],
        *(e for e in entries if e[0] in {"elo only (raw)", "market (de-vigged)"}),
    ]
    path = reliability_diagram(
        plotted,
        report_path("phase4_models_reliability.png"),
        title=f"Phase 4 base models — validation {VAL_SEASONS[0]}-{VAL_SEASONS[1]}",
    )
    report_written(path)


def _home_field_era(matrix: pl.DataFrame) -> None:
    """Home win rate per split and per validation season.

    Printed at the checkpoint because it is the explanation for the checkpoint's
    most surprising number. The calibrator is fitted where home teams won 58.3%
    of the time and scored where they won 51.3%, and no amount of calibrating on
    2016-2018 can know that. The de-vigged market line lands below the diagonal
    on these same seasons too, which is what says this is a regime change rather
    than a defect in our calibrator.
    """
    by_split = (
        matrix.filter(pl.col("split").is_in(["train", "calib", "val"]))
        .group_by("split")
        .agg(n=pl.len(), home_win_rate=pl.col("home_win").mean().round(4))
        .sort("split", descending=True)
    )
    by_season = (
        matrix.filter(pl.col("split") == "val")
        .group_by("season")
        .agg(n=pl.len(), home_win_rate=pl.col("home_win").mean().round(4))
        .sort("season")
    )
    print("\nhome-field advantage by split, and across the validation seasons:")
    print(by_split)
    print(by_season)


def _best(
    model_rows: Sequence[tuple[str, str]],
    losses: Sequence[float],
    among: Sequence[str] = (),
) -> tuple[str, str]:
    """Lowest validation log loss, as ``(model name, feature set name)``.

    Selection is on log loss rather than accuracy, per constraint 4: accuracy is
    the headline but not the target, and on 821 games it moves in steps of
    roughly 0.0012 while log loss moves continuously.
    """
    candidates = [
        (losses[i], name, set_name)
        for i, (name, set_name) in enumerate(model_rows)
        if not among or name in among
    ]
    _, name, set_name = min(candidates)
    return name, set_name


if __name__ == "__main__":
    main()
