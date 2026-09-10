"""Phase 7 checkpoint: the walk-forward number, and the single test touch.

Phases 1-6 were one fit, held still. This is the same system refitted every
week - base estimators on everything completed before kickoff, calibrator on the
trailing three seasons, held out of the base fit (D-26). It is the number the
project should be judged on, because it is the only one that describes what the
system does rather than what a snapshot of it scored.

**This script spends the test split.** 2022-2025 has been untouched since D-1
set the boundaries; it is read once, here, and anything found afterwards that
wants a test number cannot have one. Section 1 is the validation table, which is
where the reading gets sanity-checked; section 2 is the touch.

Five things are printed:

1. **Validation walk-forward**, recency-weighted against unweighted, against the
   Phase 5 frozen-split stack scored on the same games. That third row is D-26's
   comparability row: the decision retires the carried constraint that every
   phase trains on 2006-2015, so the old construction and the new one are put on
   identical games rather than compared across different ones.
2. **The test touch**, same table.
3. A reliability diagram over the walk-forward predictions.
4. Per-member rolling Brier, read back **out of the prediction log** rather than
   recomputed in memory - which is what proves the log can score itself.
5. Log summary: rows written, model versions, settled fraction.

`halt_if_suspicious` guards every table. Vegas hits 66-68% straight up; anything
above 72% on a time-ordered split is a leak, and the tripwire raises before the
number is reported (constraint 3).

Run::

    uv run python scripts/phase7_pipeline.py
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl

from nflpred.backtest import (
    CLOCK_COLUMN,
    WalkForward,
    effective_sample_size,
    frozen_split_reference,
    recency_weights,
    walk_forward,
)
from nflpred.config import (
    CALIB_WINDOW_SEASONS,
    FEATURE_MATRIX_GT_PATH,
    PREDICTIONS_DB,
    RECENCY_HALF_LIFE,
    TEST_SEASONS,
    TRAIN_SEASONS,
    VAL_SEASONS,
)
from nflpred.evaluate import (
    always_home,
    market_probability,
    metrics_table,
    reliability_diagram,
)
from nflpred.modeling.ensemble import ENSEMBLE_FEATURES, MEMBER_ORDER
from nflpred.predlog import config_suffixes, connect, rolling_brier, summary

from _shared import checked, report_path, report_written, require_matrix

pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(170)
pl.Config.set_fmt_str_lengths(46)

#: The Phase 5 headline **as published**, validation 2019-2021. Kept as the
#: historical figure and not as the comparison: D-31 rebuilt the quarterback
#: composite, so the same construction refitted today scores slightly
#: differently. The comparability row in each table is refitted live and is what
#: the closing section reads.
PHASE5_PUBLISHED: dict[str, float] = {"log_loss": 0.6395, "accuracy": 0.6291}

#: The de-vigged market on the same validation games. The reference ceiling.
MARKET_BAR: dict[str, float] = {"log_loss": 0.6107, "accuracy": 0.6462}

#: Labels, fixed so both tables read the same way.
WEIGHTED = "walk-forward (recency-weighted)"
UNWEIGHTED = "walk-forward (unweighted)"
FROZEN = "frozen-split stack A (Phase 5 construction)"


def main() -> None:
    require_matrix(FEATURE_MATRIX_GT_PATH)

    matrix = pl.read_parquet(FEATURE_MATRIX_GT_PATH)
    half_life = RECENCY_HALF_LIFE
    frozen = "off (unweighted)" if half_life is None else f"{half_life:g} league weeks"

    print("=" * 78)
    print("Phase 7 - walk-forward retraining, recency weighting, prediction log")
    print("=" * 78)
    print(
        f"matrix: {FEATURE_MATRIX_GT_PATH.name}   features: {len(ENSEMBLE_FEATURES)}   "
        f"calibration window: {CALIB_WINDOW_SEASONS} seasons (D-26)"
    )
    print(f"frozen recency half-life: {frozen}   (scripts/tune_recency.py)\n")

    validation, val_frozen = _section(
        matrix, VAL_SEASONS, half_life, "1. VALIDATION", spends_the_budget=False
    )
    test, test_frozen = _section(
        matrix, TEST_SEASONS, half_life, "2. THE TEST TOUCH", spends_the_budget=True
    )

    _reliability(validation, test)
    _log_report()
    _closing(validation, val_frozen, test, test_frozen)


# ------------------------------------------------------------------ sections


def _section(
    matrix: pl.DataFrame,
    seasons: tuple[int, int],
    half_life: float | None,
    heading: str,
    spends_the_budget: bool,
) -> tuple[WalkForward, np.ndarray]:
    """One walk-forward, its controls, and its comparators, as one table.

    Returns the walk and the frozen-split reference probabilities, so the
    closing section can quote the D-26 comparability row as refitted rather than
    as published (D-31 moved it).
    """
    print("\n" + "=" * 78)
    print(f"{heading} - {seasons[0]}-{seasons[1]}")
    print("=" * 78)
    if spends_the_budget:
        print(
            "This is the one reading of 2022-2025 the project gets (D-1). Every fit\n"
            "below was made on games completed before its target week kicked off, and\n"
            "the table is passed through the constraint-3 tripwire before it prints.\n"
        )

    started = time.perf_counter()
    weighted = _walk(matrix, seasons, half_life, "weighted")
    control = (
        weighted
        if half_life is None
        else _walk(matrix, seasons, None, "unweighted control")
    )
    elapsed = time.perf_counter() - started

    target = weighted.predictions
    y = weighted.y

    # The D-26 comparability row needs feature columns, which the prediction
    # frame does not carry - so the matrix rows for the same games, in the same
    # order, checked rather than assumed.
    features = matrix.filter(
        pl.col("game_id").is_in(target["game_id"].to_list())
    ).sort("gameday", "game_id")
    assert features["game_id"].to_list() == target["game_id"].to_list()
    reference = frozen_split_reference(matrix, features, ENSEMBLE_FEATURES)

    entries = [
        (WEIGHTED, y, weighted.probability()),
        (UNWEIGHTED, y, control.probability()),
        (FROZEN, y, reference),
        *[
            (f"member: {name}", y, weighted.probability(f"member_{name}"))
            for name in MEMBER_ORDER
        ],
        ("elo only (raw)", y, weighted.probability("elo_raw")),
        ("elo only (calibrated)", y, weighted.probability("elo_calibrated")),
        (
            "always pick home",
            y,
            always_home(
                len(y),
                matrix.filter(pl.col("season").is_between(*TRAIN_SEASONS))[
                    "home_win"
                ].mean(),
            ),
        ),
        (
            "market (de-vigged)",
            y,
            market_probability(
                target["home_moneyline"].to_numpy(), target["away_moneyline"].to_numpy()
            ),
        ),
    ]

    groups = ["walk-forward"] * 2 + ["frozen"] + ["member"] * len(MEMBER_ORDER)
    groups += ["comparator"] * 4

    table = (
        metrics_table(entries)
        .with_columns(pl.Series("group", groups))
        .select(["model", "group", "n", "accuracy", "log_loss", "brier"])
    )
    checked(table, entries)

    weeks = weighted.weeks
    print(
        f"{weeks.height} target weeks, {target.height:,} games, "
        f"{elapsed / 60:.1f} min to refit and score everything"
    )
    print(
        f"  first fit: {weeks['n_train'][0]:,} train + {weeks['n_calib'][0]:,} calib "
        f"({weeks['train_seasons'][0]} / {weeks['calib_seasons'][0]})"
    )
    print(
        f"  last  fit: {weeks['n_train'][-1]:,} train + {weeks['n_calib'][-1]:,} calib "
        f"({weeks['train_seasons'][-1]} / {weeks['calib_seasons'][-1]})"
    )
    print(table)

    if half_life is None:
        print(
            f"\n  '{UNWEIGHTED}' is the same run as '{WEIGHTED}': the frozen half-life "
            f"is None, so the two rows are one number reported twice rather than a\n"
            f"  comparison. That is the grid's answer, stated plainly."
        )
    else:
        _weighting_cost(matrix, weighted, half_life)

    _by_season(weighted, control, reference)
    return weighted, reference


def _walk(
    matrix: pl.DataFrame,
    seasons: tuple[int, int],
    half_life: float | None,
    label: str,
) -> WalkForward:
    def progress(index: int, total: int, week: dict) -> None:
        if index == 1 or index % 20 == 0 or index == total:
            print(
                f"    {label}: week {index}/{total} "
                f"({week['season']} w{week['week']:02d})",
                flush=True,
            )

    return walk_forward(
        matrix,
        seasons,
        ENSEMBLE_FEATURES,
        half_life=half_life,
        calib_seasons=CALIB_WINDOW_SEASONS,
        on_week=progress,
    )


def _weighting_cost(
    matrix: pl.DataFrame, result: WalkForward, half_life: float
) -> None:
    """What the weighting cost in effective sample size, at the last fit."""
    last = result.weeks.row(-1, named=True)
    completed = matrix.filter(pl.col(CLOCK_COLUMN) < last["league_week"])
    weights = recency_weights(completed, last["league_week"], half_life).to_numpy()
    print(
        f"\n  at the last fit, a half-life of {half_life:g} league weeks turns "
        f"{completed.height:,} games into an effective "
        f"{effective_sample_size(weights):,.0f}."
    )


def _by_season(
    weighted: WalkForward, control: WalkForward, reference: np.ndarray
) -> None:
    """The same three rows, per season. A single number can hide a bad year."""
    frame = weighted.predictions.with_columns(
        unweighted=pl.Series(control.probability()),
        frozen=pl.Series(reference),
    )

    rows = []
    for season in sorted(frame["season"].unique().to_list()):
        section = frame.filter(pl.col("season") == season)
        y = section["home_win"].to_numpy()
        entries = [
            ("walk-forward", y, section["ensemble"].to_numpy()),
            ("unweighted", y, section["unweighted"].to_numpy()),
            ("frozen split", y, section["frozen"].to_numpy()),
        ]
        scored = metrics_table(entries)
        rows.append(
            {
                "season": season,
                "n": section.height,
                "home_win_rate": round(float(np.mean(y)), 4),
                **{
                    f"{name} ll": scored["log_loss"][position]
                    for position, name in enumerate(("walk", "unwtd", "frozen"))
                },
            }
        )

    print("\nby season (log loss):")
    print(pl.DataFrame(rows))
    print(
        "  Home win rate is in the table because D-18 is the reason this phase "
        "exists: the frozen calibrator was fitted on a 58.3% era, and how it fares\n"
        "  season by season is a function of how far that season sits from it."
    )


def _reliability(validation: WalkForward, test: WalkForward) -> None:
    """One diagram, both spans, against Elo and the market."""
    combined = pl.concat(
        [validation.predictions, test.predictions], how="vertical"
    ).sort("gameday", "game_id")
    y = combined["home_win"].to_numpy()

    entries = [
        ("walk-forward ensemble", y, combined["ensemble"].to_numpy()),
        ("elo only (raw)", y, combined["elo_raw"].to_numpy()),
        (
            "market (de-vigged)",
            y,
            market_probability(
                combined["home_moneyline"].to_numpy(),
                combined["away_moneyline"].to_numpy(),
            ),
        ),
    ]
    path = reliability_diagram(
        entries,
        report_path("phase7_walkforward_reliability.png"),
        title=(
            f"Phase 7 walk-forward - {VAL_SEASONS[0]}-{TEST_SEASONS[1]}, "
            f"refit weekly"
        ),
    )
    report_written(path, label="3. RELIABILITY")
    print(
        f"   {combined.height:,} games across validation and test, every point "
        f"predicted by a model fitted only on games that had already been played."
    )


def _log_report() -> None:
    """The prediction log, read back rather than recomputed.

    Deliberately reads whatever the log happens to hold rather than writing to
    it: the checkpoint's job here is to show that a week logged by
    ``python -m nflpred.predict`` can be scored from the database alone, not to
    manufacture the rows it then reads.
    """
    print("\n" + "=" * 78)
    print("4. PER-MEMBER ROLLING BRIER, FROM THE PREDICTION LOG")
    print("=" * 78)

    if not PREDICTIONS_DB.exists():
        print(
            f"  no log at {PREDICTIONS_DB.name}. Run "
            f"`python -m nflpred.predict --season 2025 --week 8` and then "
            f"`--settle` to populate it."
        )
        return

    connection = connect()
    try:
        counts = summary(connection)
        suffixes = config_suffixes(connection)
        overall, trailing = rolling_brier(connection, window=4)
    finally:
        connection.close()

    print(
        f"configurations in the log: {', '.join(suffixes) or 'none'} - reporting on "
        f"{suffixes[0] if suffixes else 'none'}, because pooling two\nconfigurations' "
        f"predictions on the same games would score a model that never existed."
    )
    if overall.is_empty():
        print(
            f"  {counts['predictions']} predictions logged, none settled yet. "
            f"Run `python -m nflpred.predict --settle`."
        )
    else:
        print("all settled games:")
        print(overall)
        print("\ntrailing 4 weeks:")
        print(trailing)
        print(
            "  Reported, never fed back into the weights - the spec's open-questions "
            "section resolved in-season item 3 that way, because the members\n"
            "  correlate at 0.959 and every weighting scheme lands within 0.0006 of "
            "every other (D-21)."
        )

    print("\n" + "=" * 78)
    print("5. LOG SUMMARY")
    print("=" * 78)
    print(
        pl.DataFrame(
            [
                {
                    "runs": counts["runs"],
                    "predictions": counts["predictions"],
                    "member_rows": counts["member_predictions"],
                    "outcomes": counts["outcomes"],
                    "settled": counts["settled"],
                    "settled_share": counts["settled_share"],
                }
            ]
        )
    )
    versions = counts["model_versions"]
    print(f"  model versions: {', '.join(versions) if versions else 'none'}")


def _closing(
    validation: WalkForward,
    val_frozen: np.ndarray,
    test: WalkForward,
    test_frozen: np.ndarray,
) -> None:
    """The bars, and what the numbers say against them."""
    def scored(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
        table = metrics_table([("x", y, p)])
        return float(table["log_loss"][0]), float(table["accuracy"][0])

    val_ll, val_acc = scored(validation.y, validation.probability())
    val_frozen_ll, val_frozen_acc = scored(validation.y, val_frozen)
    test_ll, test_acc = scored(test.y, test.probability())
    test_frozen_ll, test_frozen_acc = scored(test.y, test_frozen)

    print("\n" + "=" * 78)
    print("WHERE IT LANDS")
    print("=" * 78)
    print(
        f"  validation walk-forward     log loss {val_ll:.4f}   accuracy {val_acc:.4f}"
    )
    print(
        f"  validation frozen split     log loss {val_frozen_ll:.4f}   "
        f"accuracy {val_frozen_acc:.4f}   "
        f"(delta {val_ll - val_frozen_ll:+.4f} log loss, "
        f"{val_acc - val_frozen_acc:+.4f} accuracy)"
    )
    print(
        f"  de-vigged market            log loss {MARKET_BAR['log_loss']:.4f}   "
        f"accuracy {MARKET_BAR['accuracy']:.4f}"
    )
    print(
        f"\n  test walk-forward           log loss {test_ll:.4f}   "
        f"accuracy {test_acc:.4f}"
    )
    print(
        f"  test frozen split           log loss {test_frozen_ll:.4f}   "
        f"accuracy {test_frozen_acc:.4f}   "
        f"(delta {test_ll - test_frozen_ll:+.4f} log loss, "
        f"{test_acc - test_frozen_acc:+.4f} accuracy)"
    )
    print(
        f"\n  For reference, Phase 5 published {PHASE5_PUBLISHED['log_loss']:.4f} / "
        f"{PHASE5_PUBLISHED['accuracy']:.4f} for the frozen split on validation. The "
        f"row above is\n  the same construction refitted today, which D-31's "
        f"quarterback-composite fix moved. Compare walk-forward against the refitted "
        f"row, not\n  against the published one - the two differ in more than the "
        f"thing being measured."
    )
    print(
        "\n  D-18 said the test era would differ from validation in either direction, "
        "and that it could not be predicted which - 2022-2025 is a third\n"
        "  home-field regime, not a continuation of the second. Read the gap between "
        "the two walk-forward numbers as that, not as overfitting."
    )


if __name__ == "__main__":
    main()
