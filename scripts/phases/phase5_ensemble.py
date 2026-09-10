"""Phase 5 checkpoint: does combining the five estimators buy anything?

Phase 4's answer to "which estimator is best" was "it barely matters" — 0.0079
of log loss across ten fitted models, none clearing the Phase 3 bar. Phase 5 is
the spec's follow-up: soft voting, then stacking, against the best single model.
The spec is explicit about the failure case — *if the ensemble does not beat it,
say so plainly rather than tuning until it does* — so this script reports a
comparison rather than searching for a winner.

Five ensembles are measured, differing in exactly one thing each:

    soft vote (6)        equal-weight mean of all six members
    soft vote (5)        the same without Elo — the control isolating Elo
    stack A              prefit, **uncalibrated** members, meta-learner on calib
    stack B              prefit, **calibrated** members, meta-learner on calib
    stack C              spec-literal out-of-fold stack on train, sigmoid on calib

Stack B is the compromised row and is labelled as one: its members' sigmoids
were fitted on the same 801 games its meta-learner is fitted on. Two parameters
per member makes the effect small, not absent.

**One honesty note.** The headline stack is picked by validation log loss among
three candidates. That is a selection made *on* validation, unlike every
hyperparameter in this project (D-11, D-16), which was frozen from the training
seasons. Three candidates is a small search and the cost is bounded, but it is
real, and the eventual test-split reading has to be read knowing it. See D-19.

Everything refits from the factory (D-17); nothing is loaded. Feature set is
`CORE_FEATURES` plus Elo's probability column — the feature-count question was
settled at Phase 4 and no ensemble here is built on 29 columns. Validation is
scored, never fitted. Test is not read.

Run::

    uv run python scripts/phase5_ensemble.py
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
import polars as pl
from sklearn.base import ClassifierMixin

from nflpred.config import (
    CALIB_SEASONS,
    FEATURE_MATRIX_GT_PATH,
    MODELS_DIR,
    TRAIN_SEASONS,
    VAL_SEASONS,
)
from nflpred.evaluate import (
    accuracy,
    always_home,
    log_loss,
    market_probability,
    metrics_table,
    reliability_diagram,
)
from nflpred.features.build import CORE_FEATURES
from nflpred.modeling.base import (
    elo_platt_probability,
    elo_probability,
    fit_all,
    fit_calibrated,
    fit_elo_platt,
    home_win_probability,
    inner_estimator,
    split_frame,
)
from nflpred.modeling.ensemble import (
    CONFIDENCE_BANDS,
    ENSEMBLE_FEATURES,
    MEMBER_ORDER,
    build_members,
    confidence,
    member_probabilities,
    meta_coefficients,
    oof_stack,
    predict_records,
    prefit_stack,
    soft_vote,
)
from nflpred.modeling.store import load_models, save_models

from _shared import checked, report_path, report_written, require_matrix

pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(160)
pl.Config.set_fmt_str_lengths(44)

#: Artifact key for the headline ensemble.
ARTIFACT_KEY = "ensemble"

#: The Phase 3 headline: scaled logistic regression at C=1.0 on 16 features,
#: uncalibrated, on the unfiltered matrix. Still the bar nothing has cleared.
PHASE3_BAR: dict[str, float] = {"log_loss": 0.6377, "accuracy": 0.6352}

#: The Phase 4 headline: xgboost on `CORE_FEATURES`, sigmoid-calibrated. The bar
#: an ensemble has to clear to have earned its complexity.
PHASE4_BAR: dict[str, float | str] = {"model": "xgboost", "log_loss": 0.6398, "accuracy": 0.6352}

#: Rows are entries, not models: a `(label, y, probabilities)` triple.
Entry = tuple[str, np.ndarray, np.ndarray]

#: The three stacking variants, by the label they carry through every table.
STACK_A = "stack A (prefit, raw members)"
STACK_B = "stack B (prefit, calibrated members)*"
STACK_C = "stack C (out-of-fold, then sigmoid)"

#: The headline, pinned rather than selected (D-22). Phase 6 forced this: the
#: spec wants the meta-learner's coefficients used as SHAP-combination weights,
#: and stack C gives lightgbm **-0.578**, which would flip the sign of every
#: factor that member contributes. Stack A's six coefficients are all positive
#: and its members are prefit rather than refitted per fold — the "cleanest
#: construction" D-19 already identified while choosing the other one on a
#: 0.0001 log-loss difference it also called noise.
HEADLINE_STACK = STACK_A


def main() -> None:
    require_matrix(FEATURE_MATRIX_GT_PATH)

    matrix = pl.read_parquet(FEATURE_MATRIX_GT_PATH)
    train, calib, val = (split_frame(matrix, s) for s in ("train", "calib", "val"))
    y_val = val["home_win"].to_numpy()

    print(
        f"train {TRAIN_SEASONS[0]}-{TRAIN_SEASONS[1]}: {train.height:,}   "
        f"calibration {CALIB_SEASONS[0]}-{CALIB_SEASONS[1]}: {calib.height:,}   "
        f"validation {VAL_SEASONS[0]}-{VAL_SEASONS[1]}: {val.height:,}   (unseen split untouched)"
    )
    print(
        f"matrix: {FEATURE_MATRIX_GT_PATH.name}   "
        f"features: {len(CORE_FEATURES)} + elo_prob = {len(ENSEMBLE_FEATURES)}\n"
    )

    started = time.perf_counter()

    # ------------------------------------------------------------- members
    #
    # The five calibrated estimators are exactly Phase 4's, refitted from the
    # factory. Two member sets fall out of them: the calibrated wrappers, and
    # the bare fitted bases underneath. Stack A uses the second, so its
    # meta-learner sees raw scores and owns the calibration itself.
    models = fit_all(matrix, CORE_FEATURES)
    platt = fit_elo_platt(calib)

    calibrated_members = build_members(models, platt=platt)
    raw_members = build_members({n: inner_estimator(m) for n, m in models.items()})

    # ----------------------------------------------------------- ensembles
    vote_six = soft_vote(calibrated_members, calib)
    vote_five = soft_vote(calibrated_members[:-1], calib)
    stack_a = prefit_stack(raw_members, calib)
    stack_b = prefit_stack(calibrated_members, calib)
    stack_c = fit_calibrated(oof_stack(CORE_FEATURES), train, calib, ENSEMBLE_FEATURES)

    ensembles: dict[str, ClassifierMixin] = {
        "soft vote (6)": vote_six,
        "soft vote (5, no elo)": vote_five,
        STACK_A: stack_a,
        STACK_B: stack_b,
        STACK_C: stack_c,
    }
    stacks: dict[str, ClassifierMixin] = {
        STACK_A: stack_a,
        STACK_B: stack_b,
        STACK_C: inner_estimator(stack_c),
    }

    entries: list[Entry] = [
        (label, y_val, home_win_probability(model, val, ENSEMBLE_FEATURES))
        for label, model in ensembles.items()
    ]

    # ------------------------------------------------------------- singles
    for name in MEMBER_ORDER[:-1]:
        entries.append((name, y_val, home_win_probability(models[name], val, CORE_FEATURES)))

    elapsed = time.perf_counter() - started

    # --------------------------------------------------------- comparators
    entries.extend(
        [
            ("elo only (raw)", y_val, elo_probability(val)),
            ("elo only (calibrated)", y_val, elo_platt_probability(platt, elo_probability(val))),
            ("always pick home", y_val, always_home(val.height, train["home_win"].mean())),
            (
                "market (de-vigged)",
                y_val,
                market_probability(
                    val["home_moneyline"].to_numpy(), val["away_moneyline"].to_numpy()
                ),
            ),
        ]
    )

    table = checked(
        metrics_table(entries)
        .with_columns(
            pl.Series(
                "group", ["ensemble"] * len(ensembles) + ["single"] * 5 + ["comparator"] * 4
            )
        )
        .select(["model", "group", "n", "accuracy", "log_loss", "brier"]),
        entries,
    )

    print(f"validation {VAL_SEASONS[0]}-{VAL_SEASONS[1]}   ({elapsed:.1f}s to fit everything)")
    print(table)
    print(
        f"\n* stack B double-dips: its members' sigmoids and its meta-learner were "
        f"fitted on the same {calib.height} games. Read it as a diagnostic, not a peer of A."
    )
    print(
        f"stack C dropped the earliest out-of-fold block: "
        f"{stacks[STACK_C].n_oof_:,} of {train.height:,} training games reached its meta-learner."
    )
    print(
        f"\nPhase 3 bar: log loss {PHASE3_BAR['log_loss']:.4f}   |   "
        f"Phase 4 bar ({PHASE4_BAR['model']}, {len(CORE_FEATURES)} features): "
        f"log loss {PHASE4_BAR['log_loss']:.4f}"
    )

    # -------------------------------------------------- headline selection
    losses = dict(zip(table["model"], table["log_loss"], strict=True))
    headline_label = HEADLINE_STACK
    best_single = min(MEMBER_ORDER[:-1], key=lambda name: losses[name])
    headline = ensembles[headline_label]
    delta = losses[headline_label] - float(PHASE4_BAR["log_loss"])

    print(
        f"\nheadline stack: {headline_label}  (pinned on construction, not on the "
        f"validation number — see D-22)"
    )
    print(
        "  Its six meta-learner coefficients are all positive, where stack C gives "
        "lightgbm -0.578; a negative weight flips the sign of every factor that member\n"
        "  contributes, which is defensible arithmetic and a poor basis for an "
        "explanation weight. Stack A also holds its members prefit rather than\n"
        "  refitting them per fold. The 0.0001 of log loss between the two is noise by "
        "D-19's own words, so pinning here *removes* the D-19 selection artefact\n"
        "  rather than adding to it: the headline is no longer chosen on validation at all."
    )
    print(
        f"vs the Phase 4 bar: {losses[headline_label]:.4f} - {PHASE4_BAR['log_loss']:.4f} = "
        f"{delta:+.4f} log loss   "
        f"({'the ensemble is worse' if delta > 0 else 'the ensemble is better'})"
    )

    # ---------------------------------------------------------- diagnostics
    probabilities = member_probabilities(calibrated_members, val)
    _correlation(probabilities)
    _coefficients(stacks)

    ensemble_probability = home_win_probability(headline, val, ENSEMBLE_FEATURES)
    _by_agreement(probabilities, ensemble_probability, y_val, val)
    _by_confidence(ensemble_probability, y_val)
    _sample_records(headline, calibrated_members, val)

    # ------------------------------------------------------------- diagram
    plotted: list[Entry] = [
        (headline_label, y_val, ensemble_probability),
        *(e for e in entries if e[0] in {best_single, "elo only (raw)", "market (de-vigged)"}),
    ]
    path = reliability_diagram(
        plotted,
        report_path("phase5_ensemble_reliability.png"),
        title=f"Phase 5 ensemble — validation {VAL_SEASONS[0]}-{VAL_SEASONS[1]}",
    )
    report_written(path)

    # ------------------------------------------------------------ artifact
    directory = save_models(
        {ARTIFACT_KEY: headline},
        ARTIFACT_KEY,
        matrix,
        ENSEMBLE_FEATURES,
        meta={"phase": 5, "variant": headline_label, "members": list(MEMBER_ORDER)},
        matrix_path=FEATURE_MATRIX_GT_PATH,
    )
    load_models(ARTIFACT_KEY, matrix)
    print(
        f"wrote the headline ensemble -> "
        f"{directory.relative_to(MODELS_DIR.parent)}  (fingerprint verified)"
    )


def _correlation(probabilities: dict[str, np.ndarray]) -> None:
    """Pairwise correlation of the six members' validation probabilities.

    The explanation for whatever the headline number turns out to be, in either
    direction. Averaging six probabilities that correlate at 0.95+ is close to a
    shrink toward their mean, and no meta-learner can extract a diversity that
    is not there — so this table is reported whatever it says.
    """
    names = list(probabilities)
    matrix = np.corrcoef(np.vstack([probabilities[n] for n in names]))
    frame = pl.DataFrame({"member": names}).with_columns(
        [pl.Series(name, matrix[:, j].round(3)) for j, name in enumerate(names)]
    )
    off_diagonal = matrix[~np.eye(len(names), dtype=bool)]
    print("\nmember correlation on validation:")
    print(frame)
    print(
        f"  off-diagonal: min {off_diagonal.min():.3f}   "
        f"mean {off_diagonal.mean():.3f}   max {off_diagonal.max():.3f}"
    )


def _coefficients(stacks: dict[str, ClassifierMixin]) -> None:
    """Meta-learner weights per stack, in member order.

    Phase 6 combines per-model SHAP values with these, so they are printed
    rather than left inside a pickle. A negative weight is called out: it means
    the meta-learner is using that member as a *contrarian* signal, which is
    defensible arithmetic and a poor basis for an explanation weight.
    """
    rows = []
    for label, stack in stacks.items():
        coefficients = meta_coefficients(stack)
        rows.append({"stack": label, **{k: round(v, 3) for k, v in coefficients.items()}})
    frame = pl.DataFrame(rows)
    print("\nmeta-learner coefficients (member order):")
    print(frame)

    negative = [
        f"{row['stack']}: {name}"
        for row in rows
        for name in MEMBER_ORDER
        if name in row and row[name] < 0
    ]
    print(
        "  negative weights: " + ("; ".join(negative) if negative else "none")
        + "  — a negative weight means the meta-learner reads that member as a contrarian signal."
    )


def _by_agreement(
    probabilities: dict[str, np.ndarray],
    ensemble_probability: np.ndarray,
    y_true: np.ndarray,
    frame: pl.DataFrame,
) -> None:
    """Accuracy and log loss by how many of the six back the ensemble's pick.

    Feeds Phase 6's "model agreement" confidence driver, which the spec asks to
    report separately from signal strength. If a 6/6 game is no more likely to
    be right than a 4/6 one, agreement is not a confidence driver here and the
    explanation layer should not pretend otherwise.
    """
    picks = np.vstack([(probabilities[name] >= 0.5) for name in probabilities])
    ensemble_pick = ensemble_probability >= 0.5
    agreed = (picks == ensemble_pick).sum(axis=0)

    rows = []
    for level in sorted(set(agreed.tolist()), reverse=True):
        mask = agreed == level
        rows.append(
            {
                "agreement": f"{level}/{len(probabilities)}",
                "n": int(mask.sum()),
                "accuracy": round(accuracy(y_true[mask], ensemble_probability[mask]), 4),
                "log_loss": round(log_loss(y_true[mask], ensemble_probability[mask]), 4),
                "mean_p": round(float(ensemble_probability[mask].mean()), 4),
            }
        )
    print(f"\nby model agreement ({frame.height:,} validation games):")
    print(pl.DataFrame(rows))
    print(
        "  Votes are the canonical six — the five calibrated estimators plus Platt-scaled "
        "Elo — which is what the output contract reports, whichever variant the\n"
        "  headline stack turns out to be. A stack holding uncalibrated members can "
        "therefore land on the other side of 0.50 from most of them (D-18 shifts a\n"
        "  sigmoid-calibrated probability up by ~0.015), which is what the low-agreement "
        "rows are made of."
    )


def _by_confidence(ensemble_probability: np.ndarray, y_true: np.ndarray) -> None:
    """Count and hit rate per a priori confidence band.

    The bands were fixed before the run (`CONFIDENCE_BANDS`), so this is a check
    rather than a fit: a band whose hit rate is not above the band below it is a
    band that is not measuring what its label claims.
    """
    labels = np.array([confidence(p) for p in ensemble_probability])
    rows = []
    for _, label in CONFIDENCE_BANDS:
        mask = labels == label
        rows.append(
            {
                "confidence": label,
                "n": int(mask.sum()),
                "share": round(float(mask.mean()), 3),
                "accuracy": round(accuracy(y_true[mask], ensemble_probability[mask]), 4)
                if mask.any()
                else None,
                "log_loss": round(log_loss(y_true[mask], ensemble_probability[mask]), 4)
                if mask.any()
                else None,
            }
        )
    print("\nby a priori confidence band (|p - 0.50| cuts at 0.03 / 0.07 / 0.13):")
    print(pl.DataFrame(rows))


def _sample_records(
    headline: ClassifierMixin, members: Sequence[tuple[str, ClassifierMixin]], val: pl.DataFrame
) -> None:
    """Two per-game records, to show the output contract's shape.

    The spec's record minus the ``explanation`` key, which is Phase 6's job.
    Printed rather than written: this is a shape check at a checkpoint, not a
    deliverable.
    """
    import json

    records = predict_records(headline, members, val.head(2))
    print("\nper-game record skeleton (explanation key is Phase 6's):")
    for record in records:
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
