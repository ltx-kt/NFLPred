"""Why the ensemble picked what it picked - SHAP attribution, in plain English.

The README's first sentence promises a system that predicts winners *and*
explains each pick. Phases 2-5 built the prediction half. This is the other one.

It arrives at an awkward moment, and that shapes what it is. Phase 5 found the
ensemble worth 0.0004 of log loss over the best single model, because the six
members' validation probabilities correlate at a mean of 0.959 (D-21). The
system's accuracy is 0.63 against a market at 0.6462. So this module is not
decorating a strong model - it is the part that tells a user **how thin the read
is**, which most weeks is the honest headline. The spec's honesty constraints
are the requirement here, not the caveat.

**What is explained, and in which space.** `TreeExplainer` explains the
*uncalibrated inner* estimator, reached through
:func:`~nflpred.modeling.base.inner_estimator`. Nothing here sums to the calibrated
number the system reports, and pretending otherwise would be the easiest lie to
tell. Each member is attributed in its own natural output space:

    logreg       analytic coef_j * (x_j - mean_j) / scale_j     log-odds
    random f.    shap.TreeExplainer                             probability
    xgboost      shap.TreeExplainer                             log-odds margin
    lightgbm     shap.TreeExplainer                             log-odds margin
    catboost     shap.TreeExplainer                             log-odds margin
    elo          none - the rating *is* the explanation         probability

logreg is analytic rather than `LinearExplainer` because the spec permits it
("or just coefficient x standardized value") and because with the
`StandardScaler` already in the `Pipeline` the contribution is exact and needs
no background sample.

**Combining into one ensemble attribution.** The spec asks for per-model SHAP
combined by the meta-learner's coefficients, "labelled explicitly as an
approximation". The naive chain rule (``coef_m * p_m(1-p_m) * shap_mj``) leaves
a Taylor error in the total, because SHAP does not hand back a derivative - it
hands back an exact finite difference. The **secant** slope removes it::

    member m has base b_m and total attribution  S_m = sum_j shap_mj
        margin-space member:  p0_m = sigmoid(b_m),  p_m = sigmoid(b_m + S_m)
        prob-space member:    p0_m = b_m,           p_m = b_m + S_m
    slope_m = (p_m - p0_m) / S_m                    (0 when S_m ~ 0)
    contribution_j = sum_m coef_m * slope_m * shap_mj

Because the meta-learner is **linear in the member probabilities**, this sums
exactly::

    sum_j contribution_j == logit(p_ensemble) - (intercept + sum_m coef_m * p0_m)

What remains an approximation - and is labelled as one in every record - is the
*allocation within a member*: a constant secant slope spreads the sigmoid's
curvature evenly across that member's features. `scripts/phases/phase6_explanations.py`
prints the reconciliation residual so the claim is a number, not a disclaimer.

**The one merge.** ``elo_diff`` and ``elo_prob`` are the same information under
two names - `nflpred.features.build` says so outright, and the two differ only in
whether home-field advantage is folded in. The five sklearn members see
``elo_diff``; the Elo member *is* ``elo_prob``. Ranked separately they land in
the top three together in three games out of four, so the narrative would read
"Elo edge, and also Elo edge". :data:`FEATURE_GLOSSARY` gives them one ``group``
and factors are ranked by group sum. Summing two contributions preserves the
additivity above exactly; the split stays visible in the per-model layer the
spec requires underneath. See D-25.

**Which member probability the slope is taken at.** The headline stack holds
*raw* members (`scripts/phases/phase5_ensemble.py` builds it from `inner_estimator`),
so its meta-learner consumed uncalibrated probabilities. The per-game record
reports the *calibrated* votes instead, and D-18 puts ~0.015 between them. Every
slope here is therefore computed from the attribution's own base value rather
than read out of the record, and :func:`explain_game` takes votes from the
record only to name dissenters - never to do arithmetic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import polars as pl

from nflpred.explain.attribution import (
    Attribution,
    Explainer,
    attribute_all,
    base_logit,
    combine,
    sigmoid,
)
from nflpred.explain.narrative import (
    FACTOR_FLOOR,
    NO_READ_EDGE,
    confidence_drivers,
    dissent,
    narrative,
    top_factors,
)
from nflpred.modeling.ensemble import ENSEMBLE_FEATURES, edge


def explain_game(
    record: Mapping[str, Any],
    row: Mapping[str, Any],
    contributions: np.ndarray,
    attributions: Mapping[str, Attribution],
    position: int,
    origin: float,
    probability: float,
    features: Sequence[str] = ENSEMBLE_FEATURES,
    k: int = 3,
) -> dict[str, Any]:
    """The ``explanation`` sub-dict for one game.

    Returns the sub-dict rather than a whole record on purpose:
    `nflpred.modeling.ensemble.predict_records` owns the pick and this module owns the
    reason, and merging here would put a `shap` import in the prediction path.
    :func:`explained_records` does the merge, which lands ``explanation`` last -
    exactly where the spec's example has it.

    ``probability`` is the ensemble's **unrounded** output. It is a parameter
    rather than a read of ``record["ensemble"]["home_win_prob"]`` because that
    field is rounded to four places for display, and reconciling against it
    would report a residual of ~6e-05 that measures the rounding rather than the
    attribution - burying the very thing the field exists to prove.
    """
    home, away = record["home_team"], record["away_team"]

    no_read = edge(probability) < NO_READ_EDGE
    pick_is_home = record["ensemble"]["pick"] == home
    factors_for, factors_against = (
        ([], [])
        if no_read
        else top_factors(contributions, row, home, away, features, k, pick_is_home)
    )

    elo = float(row["elo_prob"])
    quoted = row.get("market_prob")
    market = None if quoted is None or np.isnan(float(quoted)) else float(quoted)
    baselines: dict[str, float | None] = {
        "elo_only": round(elo, 4),
        "market_implied": None if market is None else round(market, 4),
    }

    drivers = confidence_drivers(probability, record, row, features)
    total = float(contributions.sum())

    return {
        "top_factors_for": factors_for,
        "top_factors_against": factors_against,
        "confidence_drivers": drivers,
        "dissent": dissent(record, attributions, position, row, features),
        "vs_baselines": baselines,
        "narrative": narrative(
            probability, record, factors_for, factors_against, drivers, baselines
        ),
        # The spec's step 3: always surface the per-model attributions underneath
        # so the combination above can be checked rather than trusted.
        "per_model": {
            name: [
                {
                    "feature": features[j],
                    "shap": round(float(attributions[name].values[position][j]), 4),
                }
                for j in np.argsort(-np.abs(attributions[name].values[position]))[:k]
                if abs(attributions[name].values[position][j]) >= FACTOR_FLOOR
            ]
            for name in attributions
        },
        "method": {
            "combination": (
                "approximation - per-model SHAP combined by the meta-learner's "
                "coefficients through a secant chain rule. The total is exact; what "
                "is approximate is how each member's share is spread across its own "
                "features, since one secant slope per member spreads the sigmoid's "
                "curvature evenly."
            ),
            "space": "ensemble log-odds, relative to a base of all-members-at-their-own-average",
            "orientation": (
                "each shap value is oriented toward the home team, which is the "
                "quantity the totals decompose; the for/against split is oriented "
                "toward the pick, so on an away pick the two run opposite in sign"
            ),
            "base_prob": round(float(sigmoid(origin)), 4),
            "reconciliation": round(abs(total - (_logit(probability) - origin)), 9),
        },
    }


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-12, 1 - 1e-12))
    return float(np.log(p / (1 - p)))


def explained_records(
    records: Sequence[dict[str, Any]],
    frame: pl.DataFrame,
    explainers: Mapping[str, Explainer],
    coefficients: Mapping[str, float],
    intercept: float,
    probability: np.ndarray,
    features: Sequence[str] = ENSEMBLE_FEATURES,
    k: int = 3,
) -> list[dict[str, Any]]:
    """`predict_records` output with the ``explanation`` key merged on.

    ``records``, ``frame`` and ``probability`` must be the same games in the
    same order, which they are when all three come from the same split frame.
    ``probability`` is the ensemble's unrounded output - see
    :func:`explain_game` for why it is not read back off the record.
    """
    x = frame.select(features).to_numpy().astype(float)
    attributions = attribute_all(explainers, x)
    contributions = combine(attributions, coefficients)
    origin = base_logit(attributions, coefficients, intercept)

    rows = frame.to_dicts()
    return [
        {
            **record,
            "explanation": explain_game(
                record,
                rows[position],
                contributions[position],
                attributions,
                position,
                origin,
                float(probability[position]),
                features,
                k,
            ),
        }
        for position, record in enumerate(records)
    ]
