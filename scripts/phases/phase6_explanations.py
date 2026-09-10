"""Phase 6 checkpoint: can the system say *why*, and is the arithmetic honest?

Phases 2-5 built the prediction half of the README's first sentence. This is the
other half. It arrives with the ensemble worth 0.0004 of log loss over the best
single model (D-21), which sets the job: not to dress up a strong model, but to
report how thin the read is. Most weeks that is the headline.

Four things are printed, in the order they should be read:

1. **The meta-learner's coefficients.** Stack A is now the headline, pinned on
   construction rather than selected on validation (D-22). All six weights are
   positive, which is what makes them usable as SHAP-combination weights at all.
2. **A reconciliation table.** Per-member additivity and the ensemble total,
   across all 821 validation games. This is the evidence for the exactness claim
   in `nflpred.explain` — if the secant derivation were wrong, or a member's space
   mislabelled, it shows up here as a residual well above machine epsilon.
3. **Global mean |SHAP| per feature per model.** Doubles as the evidence for the
   spec's "the stated top factor is not a globally ignored feature" test, and is
   worth reading against Phase 3's logistic-regression coefficients.
4. **Five hand-checked validation games**, chosen by rule rather than by hand so
   the selection is reproducible — including the most confident **wrong** pick,
   which is the one the spec cares most about. *A wrong prediction with a
   coherent explanation is a working system; a wrong prediction with an
   incoherent one means the attribution is broken.* No assertion covers that;
   it is read.

Validation, not test: the touch-once budget is not spent on a qualitative check.
Everything refits from the factory (D-17); nothing is loaded.

Run::

    uv run python scripts/phase6_explanations.py
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import polars as pl

from nflpred.config import FEATURE_MATRIX_GT_PATH, VAL_SEASONS
from nflpred.evaluate import market_probability
from nflpred.explain import (
    ADDITIVITY_ATOL,
    FEATURE_GLOSSARY,
    NO_READ_EDGE,
    Attribution,
    Explainer,
    attribute_all,
    base_logit,
    build_explainers,
    combine,
    explained_records,
    meta_terms,
    reference,
)
from nflpred.features.build import CORE_FEATURES
from nflpred.modeling.base import (
    fit_all,
    fit_elo_platt,
    home_win_probability,
    inner_estimator,
    split_frame,
)
from nflpred.modeling.ensemble import (
    ELO_MEMBER,
    ENSEMBLE_FEATURES,
    MEMBER_ORDER,
    build_members,
    predict_records,
    prefit_stack,
)

from _shared import report_path, report_written

pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(200)
pl.Config.set_fmt_str_lengths(40)


def main() -> None:
    if not FEATURE_MATRIX_GT_PATH.exists():
        msg = (
            f"{FEATURE_MATRIX_GT_PATH.name} not built. Run "
            f"`python -m nflpred.features.build --garbage-time`."
        )
        raise FileNotFoundError(msg)

    matrix = pl.read_parquet(FEATURE_MATRIX_GT_PATH)
    train, calib, val = (split_frame(matrix, s) for s in ("train", "calib", "val"))

    started = time.perf_counter()

    # The headline ensemble, rebuilt exactly as `scripts/phase5_ensemble.py`
    # builds it: stack A, whose members are the **raw** inner estimators. Those
    # are the objects the meta-learner consumed, and so the objects to explain.
    models = fit_all(matrix, CORE_FEATURES)
    raw = {name: inner_estimator(model) for name, model in models.items()}
    stack = prefit_stack(build_members(raw), calib)
    coefficients, intercept = meta_terms(stack)

    # Elo's origin is the training-split mean of its own probability, which
    # gives it the same kind of base the tree explainers report: the model's
    # expected output over the seasons it was built on.
    explainers = build_explainers(raw, float(train["elo_prob"].mean()))

    x = val.select(ENSEMBLE_FEATURES).to_numpy().astype(float)
    attributions = attribute_all(explainers, x)
    contributions = combine(attributions, coefficients)
    origin = base_logit(attributions, coefficients, intercept)

    elapsed = time.perf_counter() - started

    print(
        f"validation {VAL_SEASONS[0]}-{VAL_SEASONS[1]}: {val.height:,} games   "
        f"features: {len(ENSEMBLE_FEATURES)}   ({elapsed:.1f}s to fit and explain everything)"
    )
    print("headline: stack A (prefit, raw members) — pinned on construction, see D-22\n")

    _coefficients(coefficients, intercept, origin)
    _reconciliation(attributions, explainers, x, contributions, stack, val, origin)
    importance = _global_importance(attributions)

    # ------------------------------------------------------------- records
    scored = val.with_columns(
        market_prob=pl.Series(
            market_probability(
                val["home_moneyline"].to_numpy(), val["away_moneyline"].to_numpy()
            )
        )
    )
    probability = home_win_probability(stack, val, ENSEMBLE_FEATURES)
    records = predict_records(
        stack, build_members(models, platt=fit_elo_platt(calib)), scored
    )
    explained = explained_records(
        records, scored, explainers, coefficients, intercept, probability
    )

    _hand_check(explained, scored, probability)
    _no_read_share(probability)

    path = _importance_plot(importance)
    report_written(path, label="mean |SHAP| by feature")


def _coefficients(
    coefficients: Mapping[str, float], intercept: float, origin: float
) -> None:
    """The six weights, and the origin the contributions are measured from."""
    frame = pl.DataFrame(
        [{"member": n, "coefficient": round(coefficients[n], 4)} for n in MEMBER_ORDER]
    )
    print("meta-learner coefficients (stack A):")
    print(frame)
    negative = [n for n in MEMBER_ORDER if coefficients[n] < 0]
    print(
        f"  intercept {intercept:+.4f}   negative weights: "
        f"{', '.join(negative) if negative else 'none'}"
    )
    print(
        f"  base: every member at its own training-average output gives an ensemble "
        f"log-odds of {origin:+.4f} (p = {1 / (1 + np.exp(-origin)):.4f}). Contributions\n"
        f"  are measured from there, not from 0.50 — an average matchup already leans home."
    )


def _reconciliation(
    attributions: Mapping[str, Attribution],
    explainers: Mapping[str, Explainer],
    x: np.ndarray,
    contributions: np.ndarray,
    stack: object,
    val: pl.DataFrame,
    origin: float,
) -> None:
    """Does every attribution sum to the thing it claims to explain?

    The load-bearing table of this checkpoint. A member that fails here is a
    member whose explainer is pointed at the wrong object; an ensemble total
    that fails means the secant derivation is wrong or a space is mislabelled.
    """
    elo_column = len(ENSEMBLE_FEATURES) - 1
    rows = []
    for name in MEMBER_ORDER:
        attribution = attributions[name]
        if name == ELO_MEMBER:
            target = x[:, elo_column]
        else:
            target = reference(name, explainers[name].estimator, x[:, :elo_column])
        residual = np.abs(attribution.totals() + attribution.base - target)
        rows.append(
            {
                "member": name,
                "space": attribution.space,
                "base": round(attribution.base, 5),
                "max_abs_err": residual.max(),
                "mean_abs_err": residual.mean(),
                "tolerance": ADDITIVITY_ATOL.get(name, 0.0),
            }
        )

    probability = home_win_probability(stack, val, ENSEMBLE_FEATURES)
    logit = np.log(probability / (1.0 - probability))
    total = np.abs(contributions.sum(axis=1) - (logit - origin))
    rows.append(
        {
            "member": "ENSEMBLE (total)",
            "space": "logit",
            "base": round(origin, 5),
            "max_abs_err": total.max(),
            "mean_abs_err": total.mean(),
            "tolerance": 1e-6,
        }
    )

    print(f"\nreconciliation over all {val.height:,} validation games:")
    print(
        pl.DataFrame(rows).with_columns(
            pl.col("max_abs_err").map_elements(lambda v: f"{v:.2e}", return_dtype=pl.String),
            pl.col("mean_abs_err").map_elements(lambda v: f"{v:.2e}", return_dtype=pl.String),
            pl.col("tolerance").map_elements(lambda v: f"{v:.0e}", return_dtype=pl.String),
        )
    )
    print(
        "  xgboost is the loose row because its contributions come back float32; every\n"
        "  other member reconciles at machine epsilon. The ensemble total is exact by\n"
        "  construction — its residual is the float round-trip through sigmoid and back,\n"
        "  not method error, which is why the secant slope is worth the derivation."
    )


def _global_importance(attributions: Mapping[str, Attribution]) -> pl.DataFrame:
    """Mean |SHAP| per feature per model, over the whole validation split.

    Two things to read here. `elo_diff` should dominate — it is the largest
    single input to five of six members, and if a tree model ranks something
    else first that is worth a look rather than a shrug. And a feature that is
    **exactly** zero for a model is one that model never splits on: the boosters
    are minimum-capacity by D-16 (xgboost is a stump ensemble, catboost depth 2,
    lightgbm 4 leaves), so a handful of untouched columns is expected geometry
    rather than a bug.
    """
    frame = pl.DataFrame(
        {"feature": list(ENSEMBLE_FEATURES)}
        | {
            name: np.abs(attributions[name].values).mean(axis=0).round(4)
            for name in MEMBER_ORDER
        }
    ).with_columns(
        total=pl.sum_horizontal([pl.col(n) for n in MEMBER_ORDER]).round(4)
    ).sort("total", descending=True)

    print("\nglobal mean |SHAP| per feature per model (validation, own space):")
    print(frame)

    # A pair is *visible* when the member can actually see the column: Elo sees
    # only `elo_prob`, the five sklearn members see only the other 16. Anything
    # else is column layout rather than indifference and must not be reported as
    # a model ignoring a feature.
    ignored = [
        f"{name}/{feature}"
        for name in MEMBER_ORDER
        for feature, value in zip(frame["feature"], frame[name], strict=True)
        if value == 0.0 and (name == ELO_MEMBER) == (feature == "elo_prob")
    ]
    print(
        "  features a model never splits on: "
        + (", ".join(ignored) if ignored else "none")
        + "\n  (the five sklearn members cannot see elo_prob and Elo cannot see the other 16 —"
        "\n  that is column layout, not indifference, and is excluded from the list above.)"
    )
    return frame


def _hand_check(
    explained: Sequence[Mapping[str, object]], val: pl.DataFrame, probability: np.ndarray
) -> None:
    """Five validation games, chosen by rule so the selection is reproducible.

    The spec asks for five hand-checked games including at least one the model
    got wrong. Picking them by eye would make the checkpoint unreproducible and
    would let a flattering sample in, so each is the argmax of a stated rule.
    """
    y = val["home_win"].to_numpy()
    edge = np.abs(probability - 0.5)
    correct = (probability >= 0.5) == (y >= 0.5)
    agreement = np.array([int(r["agreement"].split("/")[0]) for r in explained])
    backup = (
        (val["home_backup_qb_starting"].to_numpy() == 1)
        | (val["away_backup_qb_starting"].to_numpy() == 1)
    )

    picks = {
        "most confident CORRECT pick": np.argmax(np.where(correct, edge, -1.0)),
        "most confident WRONG pick — the one the spec cares about": np.argmax(
            np.where(~correct, edge, -1.0)
        ),
        "a 5/6 member disagreement": np.argmax(np.where(agreement == 5, edge, -1.0)),
        "a sub-0.03 no-read game": np.argmin(edge),
        "a game with a non-usual starting quarterback": np.argmax(
            np.where(backup, edge, -1.0)
        ),
    }

    print("\n" + "=" * 78)
    print("five hand-checked validation games, chosen by rule")
    print("=" * 78)
    for label, position in picks.items():
        index = int(position)
        record = explained[index]
        outcome = "tie" if y[index] == 0.5 else ("hit" if correct[index] else "MISS")
        print(f"\n--- {label}  [{outcome}, actual home_win={y[index]:g}] ---")
        print(json.dumps(record, indent=2))


def _no_read_share(probability: np.ndarray) -> None:
    """How often the system declines to have an opinion.

    The honesty constraint, as a rate. If nothing lands inside the band the
    constraint is not wired up; roughly one game a week should refuse to explain
    itself.
    """
    inside = np.abs(probability - 0.5) < NO_READ_EDGE
    print(
        f"\nno-read games: {inside.sum()} of {len(probability)} "
        f"({inside.mean():.1%}) fall within {NO_READ_EDGE:.2f} of a coin flip and "
        f"produce no ranked rationale."
    )


def _importance_plot(importance: pl.DataFrame, top: int = 12) -> Path:
    """Mean |SHAP| by feature, one series per model.

    Follows `nflpred.evaluate.reliability_diagram`'s conventions — deferred import,
    ``Agg``, return the `Path` — so the two report generators behave the same.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = importance.head(top).reverse()
    labels = frame["feature"].to_list()
    positions = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 6))
    left = np.zeros(len(labels))
    for name in MEMBER_ORDER:
        widths = np.asarray(frame[name].to_list(), dtype=float)
        ax.barh(positions, widths, left=left, label=name, height=0.72)
        left += widths

    ax.set_yticks(positions, labels, fontsize=8)
    ax.set_xlabel("mean |SHAP| (each model in its own output space, stacked)")
    ax.set_title(
        f"Phase 6 attribution — validation {VAL_SEASONS[0]}-{VAL_SEASONS[1]}"
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    path = report_path("phase6_explanation_importance.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _glossary_sanity() -> None:
    """Fail early if a feature set could put a raw column name in front of a user."""
    missing = [name for name in ENSEMBLE_FEATURES if name not in FEATURE_GLOSSARY]
    if missing:
        msg = f"FEATURE_GLOSSARY is missing {missing}"
        raise KeyError(msg)


if __name__ == "__main__":
    _glossary_sanity()
    main()
