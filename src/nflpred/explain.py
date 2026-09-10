"""Why the ensemble picked what it picked — SHAP attribution, in plain English.

The README's first sentence promises a system that predicts winners *and*
explains each pick. Phases 2-5 built the prediction half. This is the other one.

It arrives at an awkward moment, and that shapes what it is. Phase 5 found the
ensemble worth 0.0004 of log loss over the best single model, because the six
members' validation probabilities correlate at a mean of 0.959 (D-21). The
system's accuracy is 0.63 against a market at 0.6462. So this module is not
decorating a strong model — it is the part that tells a user **how thin the read
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
    elo          none — the rating *is* the explanation         probability

logreg is analytic rather than `LinearExplainer` because the spec permits it
("or just coefficient x standardized value") and because with the
`StandardScaler` already in the `Pipeline` the contribution is exact and needs
no background sample.

**Combining into one ensemble attribution.** The spec asks for per-model SHAP
combined by the meta-learner's coefficients, "labelled explicitly as an
approximation". The naive chain rule (``coef_m * p_m(1-p_m) * shap_mj``) leaves
a Taylor error in the total, because SHAP does not hand back a derivative — it
hands back an exact finite difference. The **secant** slope removes it::

    member m has base b_m and total attribution  S_m = sum_j shap_mj
        margin-space member:  p0_m = sigmoid(b_m),  p_m = sigmoid(b_m + S_m)
        prob-space member:    p0_m = b_m,           p_m = b_m + S_m
    slope_m = (p_m - p0_m) / S_m                    (0 when S_m ~ 0)
    contribution_j = sum_m coef_m * slope_m * shap_mj

Because the meta-learner is **linear in the member probabilities**, this sums
exactly::

    sum_j contribution_j == logit(p_ensemble) - (intercept + sum_m coef_m * p0_m)

What remains an approximation — and is labelled as one in every record — is the
*allocation within a member*: a constant secant slope spreads the sigmoid's
curvature evenly across that member's features. `scripts/phase6_explanations.py`
prints the reconciliation residual so the claim is a number, not a disclaimer.

**The one merge.** ``elo_diff`` and ``elo_prob`` are the same information under
two names — `nflpred.features.build` says so outright, and the two differ only in
whether home-field advantage is folded in. The five sklearn members see
``elo_diff``; the Elo member *is* ``elo_prob``. Ranked separately they land in
the top three together in three games out of four, so the narrative would read
"Elo edge, and also Elo edge". :data:`FEATURE_GLOSSARY` gives them one ``group``
and factors are ranked by group sum. Summing two contributions preserves the
additivity above exactly; the split stays visible in the per-model layer the
spec requires underneath. See D-25.

**Which member probability the slope is taken at.** The headline stack holds
*raw* members (`scripts/phase5_ensemble.py` builds it from `inner_estimator`),
so its meta-learner consumed uncalibrated probabilities. The per-game record
reports the *calibrated* votes instead, and D-18 puts ~0.015 between them. Every
slope here is therefore computed from the attribution's own base value rather
than read out of the record, and :func:`explain_game` takes votes from the
record only to name dissenters — never to do arithmetic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import polars as pl
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline

from nflpred.modeling.ensemble import (
    CONFIDENCE_BANDS,
    ELO_MEMBER,
    ENSEMBLE_FEATURES,
    MEMBER_ORDER,
    meta_coefficients,
)

# ---------------------------------------------------------------- constants

#: Attribution spaces a member can live in.
MARGIN: Final[str] = "margin"
PROBABILITY: Final[str] = "probability"

#: Below this the secant slope is undefined — and the member's probability has
#: not moved either, so its contribution is zero rather than a divide-by-zero.
_FLAT: Final[float] = 1e-12

#: A contribution smaller than this in log-odds is not a factor worth speaking.
#: Roughly 0.0025 of probability at the middle of the range — below that, naming
#: something as a reason overstates it. Set high enough that a lopsided game can
#: legitimately return an **empty** "against" list rather than scraping one up:
#: on a 6/6 blowout there is genuinely nothing cutting the other way, and saying
#: so is the honest output.
FACTOR_FLOOR: Final[float] = 0.01

#: The "no meaningful read" cut, in distance from 0.50. Deliberately *derived*
#: from the confidence bands rather than restated: the spec's honesty constraint
#: ("within ~0.03 of 0.50, say the model has no read") and the band labelled
#: "no read" are the same threshold, and two constants that must agree are one
#: constant with an extra chance to disagree.
NO_READ_EDGE: Final[float] = CONFIDENCE_BANDS[0][0]

#: Distance from an anchor beyond which a confident pick is *surfaced* as a
#: disagreement rather than smoothed over, per the spec's third honesty rule.
ANCHOR_GAP: Final[float] = 0.15

#: Language that would turn a statement about model behaviour into a claim about
#: football causation. SHAP describes what the model weights, never what wins
#: games. `tests/test_explanations.py` asserts none of these appears in any
#: generated string; the list lives here because it is a statement of the
#: constraint, and there it is only an assertion.
BANNED_PHRASES: Final[tuple[str, ...]] = (
    "causes",
    "caused",
    "causing",
    "because",
    "due to",
    "leads to",
    "led to",
    "results in",
    "resulted in",
    "guarantees",
    "ensures",
    "will win",
    "should win",
    "proves",
)


# ----------------------------------------------------------- the glossary

@dataclass(frozen=True)
class Feature:
    """One column, in language a reader who does not know the schema can use.

    ``favours`` is the sign convention and it is **not** uniform, which is the
    whole reason it is written down. Most columns are built ``home - away`` over
    a quantity where more is better, so a positive value favours the home team.
    Five are not, and each one is a chance to emit confidently backwards prose:

    * ``net_sack_r8_diff`` nets sacks *allowed* against sacks *generated*, so
      positive is bad for the home team (train correlation with the target:
      -0.18).
    * ``turnover_diff_shrunk`` inherits ``net_turnovers``, which
      `nflpred.features.build` builds as a turnover *deficit* (-0.07).
    * ``away_pass_epa_matchup_r8`` / ``away_rush_epa_matchup_r8`` are the away
      offence against the home defence, so positive is bad for the home team.
    * ``home_backup_qb_starting`` at 1 is bad for the home team while
      ``away_backup_qb_starting`` at 1 is good for it — deliberately asymmetric
      columns, and differencing them would have said they were the same event.

    ``div_game``, ``is_dome``, ``is_neutral_site`` and ``wind`` favour nobody and
    must never be phrased as favouring anybody.

    ``kind`` decides how a value is spoken: a ``diff`` names whichever team the
    edge belongs to, a ``flag`` is only ever spoken when it fired (see
    :func:`top_factors`), a ``level`` describes the fixture.

    ``group`` is the ranking key. It defaults to the column's own name; the one
    place it does not is Elo, whose two columns are one factor.
    """

    plain: str
    unit: str
    favours: str
    kind: str
    group: str = ""

    def key(self, name: str) -> str:
        return self.group or name


def _diff(plain: str, unit: str, favours: str = "home", group: str = "") -> Feature:
    return Feature(plain, unit, favours, "diff", group)


def _flag(plain: str, favours: str) -> Feature:
    return Feature(plain, "flag", favours, "flag")


#: Every column any defined feature set can put in front of a user, in language
#: that does not leak a schema name. Covers `PHASE3_FEATURES` (29) plus
#: ``elo_prob``, which is a superset of `CORE_FEATURES` and therefore of
#: :data:`~nflpred.modeling.ensemble.ENSEMBLE_FEATURES`. The spec asks this to fail
#: loudly on a missing entry rather than degrade, so
#: `tests/test_explanations.py` asserts completeness rather than trusting it.
#:
#: Two things every phrase here is careful about, both measured rather than
#: assumed:
#:
#: * The matrix in use is the **garbage-time-filtered** one (D-9), so every rate
#:   is over competitive plays, not all plays. "over the last 8 games" is a
#:   deliberate simplification and the docstring is where the asterisk lives.
#: * Rolling windows **cross season boundaries** (D-6), so in week 1 the entire
#:   window is last season. :func:`confidence_drivers` says so out loud rather
#:   than letting the phrase imply this season's form.
FEATURE_GLOSSARY: Final[dict[str, Feature]] = {
    # --- team form, net differentials (own production minus what was allowed)
    "net_epa_r4_diff": _diff("net EPA per play over the last 4 games", "epa"),
    "net_epa_r8_diff": _diff("net EPA per play over the last 8 games", "epa"),
    "net_epa_std_diff": _diff("net EPA per play this season", "epa"),
    "net_success_r8_diff": _diff("net success rate over the last 8 games", "pct"),
    "net_pass_epa_r8_diff": _diff("net passing EPA per play over the last 8 games", "epa"),
    "net_rush_epa_r8_diff": _diff("net rushing EPA per play over the last 8 games", "epa"),
    "net_yards_per_play_r8_diff": _diff("net yards per play over the last 8 games", "yards"),
    "net_explosive_r8_diff": _diff(
        "net rate of 20-yard plays over the last 8 games", "pct"
    ),
    "net_points_r8_diff": _diff("net points per game over the last 8 games", "points"),
    "net_third_down_r8_diff": _diff("net third-down rate over the last 8 games", "pct"),
    # Sacks allowed minus sacks generated: positive is the bad direction.
    "net_sack_r8_diff": _diff(
        "net sack rate over the last 8 games", "pct", favours="away"
    ),
    # --- the same pass/rush signal, offence and defence kept apart
    "home_pass_epa_matchup_r8": _diff(
        "passing EPA per play against the opposing pass defence", "epa"
    ),
    "away_pass_epa_matchup_r8": _diff(
        "passing EPA per play against the opposing pass defence", "epa", favours="away"
    ),
    "home_rush_epa_matchup_r8": _diff(
        "rushing EPA per play against the opposing run defence", "epa"
    ),
    "away_rush_epa_matchup_r8": _diff(
        "rushing EPA per play against the opposing run defence", "epa", favours="away"
    ),
    # --- Tier 1. Both Elo columns are one factor; see the module docstring.
    "elo_diff": _diff("Elo rating", "elo", group="elo"),
    "elo_prob": Feature(
        "Elo's own win probability, home field included", "prob", "home", "level", "elo"
    ),
    "qb_epa_r4_diff": _diff("quarterback EPA per dropback over the last 4 games", "epa"),
    "qb_epa_r8_diff": _diff("quarterback EPA per dropback over the last 8 games", "epa"),
    "qb_cpoe_r8_diff": _diff(
        "quarterback completion percentage over expected, last 8 games", "cpoe"
    ),
    # Not "a backup": the flag fires when the projected starter is not the modal
    # starter of the last 8 games, which is also true of a starter returning
    # from injury or a new franchise quarterback. It fires on ~19% of games, so
    # a reader who knows the sport will notice if this is overstated.
    "home_backup_qb_starting": _flag(
        "is starting a quarterback other than its usual starter", "away"
    ),
    "away_backup_qb_starting": _flag(
        "is starting a quarterback other than its usual starter", "home"
    ),
    # --- situational
    "rest_diff": _diff("days of rest", "days"),
    "short_week_diff": _diff("short-week scheduling", "count"),
    "off_bye_diff": _diff("bye-week rest", "count"),
    "div_game": _flag("a divisional matchup", "neither"),
    "is_dome": _flag("an indoor game", "neither"),
    "is_neutral_site": _flag("a neutral-site game", "neither"),
    "wind": Feature("wind at kickoff", "mph", "neither", "level"),
    # --- Tier 3, already shrunk to the seventh of it that persists
    "turnover_diff_shrunk": _diff(
        "recent turnover margin, shrunk to the share that persists",
        "turnovers",
        favours="away",
    ),
}

#: Formatters, one per ``unit``. A magnitude is always spoken unsigned — the
#: team it belongs to is named instead, which is unambiguous where a bare
#: "-0.14 for Buffalo" is not.
_FORMATS: Final[dict[str, str]] = {
    "epa": "{v:.3f} EPA/play",
    "pct": "{v:.1f} percentage points",
    "yards": "{v:.2f} yards/play",
    "points": "{v:.1f} points/game",
    "elo": "{v:.0f} Elo points",
    "cpoe": "{v:.1f} CPOE",
    "days": "{v:.0f} days",
    "count": "{v:.0f}",
    "turnovers": "{v:.2f} turnovers/game",
    "prob": "{v:.0%}",
    "mph": "{v:.0f} mph",
}

#: Units whose natural reading is percentage points rather than a fraction.
_SCALE_100: Final[frozenset[str]] = frozenset({"pct"})


def format_value(unit: str, value: float) -> str:
    """One feature value, in its own units, unsigned."""
    magnitude = abs(float(value))
    if unit in _SCALE_100:
        magnitude *= 100.0
    return _FORMATS[unit].format(v=magnitude)


def describe(name: str, value: float, home: str, away: str) -> str:
    """A plain-English phrase for one feature at one value.

    Never returns a schema name. A ``diff`` names the team the edge belongs to,
    which is what makes the :data:`Feature.favours` convention visible in the
    output and therefore testable.
    """
    feature = FEATURE_GLOSSARY[name]
    if feature.kind == "flag":
        if feature.favours == "neither":
            return feature.plain
        team = home if name.startswith("home_") else away
        return f"{team} {feature.plain}"
    if feature.kind == "level":
        if name == "elo_prob":
            return f"{feature.plain} ({float(value):.0%} for {home})"
        return f"{format_value(feature.unit, value)} of {feature.plain}"

    beneficiary = home if (float(value) >= 0) == (feature.favours == "home") else away
    return f"{beneficiary}'s edge in {feature.plain} ({format_value(feature.unit, value)})"


# --------------------------------------------------------- the attributions

@dataclass(frozen=True)
class Attribution:
    """One member's per-feature contributions, padded to the 17-column layout.

    ``values`` is ``(n, 17)``: the five sklearn members were fitted on the first
    16 columns and carry a zero in the ``elo_prob`` slot; the Elo member is the
    mirror image. Padding here rather than at each call site is what lets
    :func:`combine` be a plain weighted sum over one common axis.
    """

    values: np.ndarray
    base: float
    space: str

    def totals(self) -> np.ndarray:
        return self.values.sum(axis=1)

    def probabilities(self) -> tuple[np.ndarray, float]:
        """``(p_after, p_base)`` — where this member's output sits, and its origin."""
        if self.space == MARGIN:
            return _sigmoid(self.base + self.totals()), float(_sigmoid(self.base))
        return self.base + self.totals(), float(self.base)


@dataclass(frozen=True)
class Explainer:
    """How one member gets attributed. Built once per run; that is the slow part."""

    name: str
    estimator: ClassifierMixin | None
    shap: Any | None
    space: str
    base: float


def _sigmoid(z: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def reference(name: str, estimator: ClassifierMixin, x: np.ndarray) -> np.ndarray:
    """The raw output an attribution must sum to, per member, in its own space.

    This is the additivity target, and it is deliberately each library's own
    uncalibrated call rather than anything routed through
    `CalibratedClassifierCV` — a member that fails to reconcile against this is
    almost always an explainer pointed at the calibrated wrapper.
    """
    if name == "logreg":
        return np.asarray(estimator.decision_function(x), dtype=float)
    if name == "random forest":
        return np.asarray(estimator.predict_proba(x)[:, 1], dtype=float)
    if name == "xgboost":
        return np.asarray(estimator.predict(x, output_margin=True), dtype=float)
    if name == "lightgbm":
        return np.asarray(estimator.predict(x, raw_score=True), dtype=float)
    if name == "catboost":
        return np.asarray(estimator.predict(x, prediction_type="RawFormulaVal"), dtype=float)
    msg = f"no additivity reference defined for member {name!r}"
    raise KeyError(msg)


#: Per-member additivity tolerance. XGBoost is the outlier and the reason this
#: is a dict rather than one constant: its contributions come back **float32**,
#: so it reconciles at ~1e-6 while every other member reconciles at ~1e-15. A
#: single tight tolerance across all five fails on xgboost alone.
ADDITIVITY_ATOL: Final[dict[str, float]] = {
    "logreg": 1e-10,
    "random forest": 1e-9,
    "xgboost": 1e-5,
    "lightgbm": 1e-9,
    "catboost": 1e-9,
}


def _candidates(
    raw: object, expected: object, n_features: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Every plausible (values, base) reading of one library's SHAP output.

    Shape is the one thing that genuinely differs between the four tree
    libraries, and it has changed across shap releases in both directions: a
    binary forest is ``(n, f, 2)`` on shap 0.45+ and was a list of two before
    it; a booster is ``(n, f)`` with a scalar base; a native contributions call
    returns ``(n, f+1)`` with the bias in the last column. Rather than encode a
    version's conventions and silently mis-slice when they move,
    :func:`_normalise` generates the readings and lets **additivity** pick.
    """
    values = getattr(raw, "values", raw)
    base = getattr(raw, "base_values", expected)

    out: list[tuple[np.ndarray, np.ndarray]] = []
    if isinstance(values, list):
        for k in (1, 0):
            if k < len(values):
                b = base[k] if isinstance(base, (list, tuple)) else np.asarray(base)
                out.append((np.asarray(values[k], dtype=float), np.asarray(b, dtype=float)))
        return out

    values = np.asarray(values, dtype=float)
    base_arr = np.asarray(base, dtype=float)
    if values.ndim == 3:
        for k in (1, 0):
            b = base_arr[..., k] if base_arr.ndim else base_arr
            out.append((values[:, :, k], b))
    elif values.ndim == 2 and values.shape[1] == n_features + 1:
        out.append((values[:, :-1], values[:, -1]))
    elif values.ndim == 2:
        out.append((values, base_arr))
    return out


def _normalise(
    name: str, raw: object, expected: object, reference: np.ndarray, n_features: int
) -> tuple[np.ndarray, float]:
    """Pick the reading of ``raw`` that actually reconciles, or fail loudly.

    Returns ``(values, base)`` with ``values.sum(1) + base == reference``. The
    base is collapsed to a scalar only after checking it is constant across
    rows — a per-row base means the explainer is doing something this module
    does not model, and quietly averaging it would hide that.
    """
    atol = ADDITIVITY_ATOL[name]
    errors: list[float] = []
    for values, base in _candidates(raw, expected, n_features):
        residual = np.abs(values.sum(axis=1) + np.ravel(base) - reference).max()
        errors.append(float(residual))
        if residual < atol:
            flat = np.ravel(base)
            if flat.size > 1 and float(np.ptp(flat)) > atol:
                msg = (
                    f"{name}: base value varies across rows by {np.ptp(flat):.3e}. "
                    f"This module assumes one expected value per member."
                )
                raise ValueError(msg)
            return values.astype(float), float(flat.flat[0])

    msg = (
        f"{name}: no reading of the SHAP output reconciles with its own raw "
        f"output (best residual {min(errors, default=float('nan')):.3e}, "
        f"tolerance {atol:.0e}). The explainer is pointed at the wrong object — "
        f"most likely the calibrated wrapper rather than inner_estimator()."
    )
    raise ValueError(msg)


def build_explainers(
    estimators: Mapping[str, ClassifierMixin], elo_base: float
) -> dict[str, Explainer]:
    """One explainer per member. Construction is the expensive part, not use.

    ``estimators`` are the **raw** inner estimators — the objects the headline
    stack's meta-learner actually consumed — keyed by project name. ``elo_base``
    is the training-split mean of ``elo_prob``, which gives the Elo member the
    same kind of origin the tree explainers use: the model's expected output
    over the seasons it was built on.

    `shap` is imported here rather than at module scope, following
    `nflpred.evaluate.reliability_diagram`'s deferred `matplotlib` import: the
    glossary is useful on its own, and importing this module should not pull in
    numba.
    """
    import shap

    explainers: dict[str, Explainer] = {}
    for name in MEMBER_ORDER:
        if name == ELO_MEMBER:
            explainers[name] = Explainer(name, None, None, PROBABILITY, elo_base)
            continue

        estimator = estimators[name]
        if name == "logreg":
            # Analytic, and exact: the scaler is already in the Pipeline, so the
            # contribution needs no background sample. The base is the
            # intercept, which sits within 4e-4 of the training-mean decision
            # value — the same kind of origin the tree explainers report.
            base = float(estimator.named_steps["lr"].intercept_[0])
            explainers[name] = Explainer(name, estimator, None, MARGIN, base)
            continue

        space = PROBABILITY if name == "random forest" else MARGIN
        # `tree_path_dependent` explicitly, and no background data: the
        # `interventional` alternative sub-samples a background set with an RNG,
        # which would make the same game explain differently from week to week.
        # The spec requires reproducible narratives, so this is not a default to
        # inherit silently.
        explainer = shap.TreeExplainer(estimator, feature_perturbation="tree_path_dependent")
        explainers[name] = Explainer(name, estimator, explainer, space, float("nan"))
    return explainers


def attribute(explainer: Explainer, x: np.ndarray) -> Attribution:
    """One member's attribution over ``x``, padded to the 17-column layout.

    ``x`` is the full ensemble matrix; the slice each member was fitted on is
    taken here. `PrefitMember` does that slicing internally at prediction time,
    but SHAP explains the *inner* estimator, which has never seen 17 columns.
    """
    x = np.asarray(x, dtype=float)
    width = len(ENSEMBLE_FEATURES)
    elo_column = width - 1

    if explainer.name == ELO_MEMBER:
        values = np.zeros((x.shape[0], width))
        values[:, elo_column] = x[:, elo_column] - explainer.base
        return Attribution(values, explainer.base, PROBABILITY)

    x16 = x[:, :elo_column]
    if explainer.name == "logreg":
        pipeline: Pipeline = explainer.estimator
        scale = pipeline.named_steps["scale"]
        coefficients = pipeline.named_steps["lr"].coef_[0]
        narrow = (x16 - scale.mean_) / scale.scale_ * coefficients
        base = explainer.base
    else:
        raw = explainer.shap(x16)
        narrow, base = _normalise(
            explainer.name,
            raw,
            getattr(explainer.shap, "expected_value", None),
            reference(explainer.name, explainer.estimator, x16),
            x16.shape[1],
        )

    values = np.zeros((x.shape[0], width))
    values[:, :elo_column] = narrow
    return Attribution(values, float(base), explainer.space)


def attribute_all(
    explainers: Mapping[str, Explainer], x: np.ndarray
) -> dict[str, Attribution]:
    """Every member's attribution, in :data:`~nflpred.modeling.ensemble.MEMBER_ORDER`."""
    return {name: attribute(explainers[name], x) for name in MEMBER_ORDER}


# ------------------------------------------------------------ the combination

def meta_terms(stack: ClassifierMixin) -> tuple[dict[str, float], float]:
    """The meta-learner's per-member weights and its intercept.

    :func:`~nflpred.modeling.ensemble.meta_coefficients` is the owner of the weights;
    the intercept has no accessor because until now nothing needed it, and it is
    read here rather than by widening that module's surface for one caller.
    """
    return meta_coefficients(stack), float(np.ravel(stack.final_estimator_.intercept_)[0])


def secant_slopes(attributions: Mapping[str, Attribution]) -> dict[str, np.ndarray]:
    """Per-member secant slope from attribution space to probability.

    The slope that makes the combination exact. A member whose attributions sum
    to ~0 has not moved its own probability either, so its slope is 0 rather
    than a divide-by-zero.
    """
    slopes: dict[str, np.ndarray] = {}
    for name, attribution in attributions.items():
        totals = attribution.totals()
        after, before = attribution.probabilities()
        slope = np.zeros_like(totals)
        moved = np.abs(totals) > _FLAT
        slope[moved] = (after[moved] - before) / totals[moved]
        slopes[name] = slope
    return slopes


def combine(
    attributions: Mapping[str, Attribution], coefficients: Mapping[str, float]
) -> np.ndarray:
    """Per-feature contributions to the ensemble's log-odds. ``(n, 17)``.

    The secant chain rule from the module docstring. Exactly additive against
    :func:`base_logit`; `tests/test_explanations.py` asserts that at 1e-9 rather
    than at a hand-waved tolerance.
    """
    slopes = secant_slopes(attributions)
    total = np.zeros_like(next(iter(attributions.values())).values)
    for name, attribution in attributions.items():
        total += coefficients[name] * slopes[name][:, None] * attribution.values
    return total


def base_logit(
    attributions: Mapping[str, Attribution],
    coefficients: Mapping[str, float],
    intercept: float,
) -> float:
    """The ensemble's log-odds when every member returns its own base output.

    The origin the contributions are measured from. Worth two caveats, stated
    rather than buried: the member bases are 2006-2015 quantities while the
    meta-learner's weights were fitted on 2016-2018, so this is a hybrid
    reference and not the ensemble's prediction on any real game; and it sits
    *above* 0.50, because an average matchup already leans home.
    """
    return float(
        intercept
        + sum(
            coefficients[name] * attribution.probabilities()[1]
            for name, attribution in attributions.items()
        )
    )


# ------------------------------------------------------------- ranked factors

def top_factors(
    contributions: np.ndarray,
    row: Mapping[str, Any],
    home: str,
    away: str,
    features: Sequence[str] = ENSEMBLE_FEATURES,
    k: int = 3,
    pick_is_home: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The spec's four-key factor dicts, ranked, split for and against the pick.

    Grouped before ranking, so Elo's two columns are one factor rather than the
    same fact stated twice (see the module docstring).

    A 0/1 flag sitting at 0 is **suppressed**: "the away team is not starting a
    different quarterback" is not a factor, it is the absence of one, and a tree
    model will happily hand it a non-zero contribution anyway.

    **Two different orientations meet here, and conflating them prints prose
    that is backwards.** A contribution is in ensemble log-odds *for the home
    team* — that is the quantity the additivity identity decomposes, so it is
    what the ``shap`` field reports. But "for" and "against" are relative to the
    **pick**, per the spec ("which features drove the pick, which cut against
    it"). On an away pick the two run opposite: a negative contribution is what
    supports the verdict. ``pick_is_home`` is what keeps them straight, and
    without it every away pick — roughly two games in five — would file its
    supporting factors under "against".
    """
    grouped: dict[str, dict[str, Any]] = {}
    for position, name in enumerate(features):
        feature = FEATURE_GLOSSARY[name]
        value = float(row[name])
        if feature.kind == "flag" and value == 0.0:
            continue

        key = feature.key(name)
        entry = grouped.setdefault(key, {"feature": name, "value": value, "shap": 0.0})
        entry["shap"] += float(contributions[position])

    ranked = sorted(grouped.values(), key=lambda e: (-abs(e["shap"]), e["feature"]))

    def emit(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "feature": e["feature"],
                "value": round(e["value"], 4),
                "shap": round(e["shap"], 4),
                "plain": describe(e["feature"], e["value"], home, away),
            }
            for e in entries[:k]
        ]

    toward = 1.0 if pick_is_home else -1.0
    keep = [e for e in ranked if abs(e["shap"]) >= FACTOR_FLOOR]
    return (
        emit([e for e in keep if e["shap"] * toward > 0]),
        emit([e for e in keep if e["shap"] * toward < 0]),
    )


# ------------------------------------------------------ confidence, decomposed

def confidence_drivers(
    probability: float,
    record: Mapping[str, Any],
    row: Mapping[str, Any],
    features: Sequence[str] = ENSEMBLE_FEATURES,
) -> dict[str, str]:
    """The spec's three drivers, separated. A single number hides all three.

    Two of the spec's three data-quality checks cannot fire on this matrix, and
    saying so is more useful than emitting a flag that is always green:

    * *"fewer than 4 games of rolling history"* — D-6 runs rolling windows
      across season boundaries, so ``games_in_window_r4`` is 4 for every row in
      the matrix. Structurally dead. The live version of the same concern is
      early-season form being mostly last season's, which is what is reported.
    * *"any feature with a null that had to be imputed"* — the 17 columns carry
      zero nulls in train, calib and validation. The check is kept, so it fires
      if that ever stops being true.

    The third fires often enough to carry the driver on its own: 269 of 821
    validation games have a different-than-usual starter on one side.
    """
    edge = abs(probability - 0.5)
    pick = record["ensemble"]["pick"]
    # Quoted for the team actually picked, not for the home team. `probability`
    # is P(home win) everywhere in this project, so on an away pick the two are
    # complements — and "favours DEN at 39.0%" is a sentence that reads as a
    # contradiction.
    for_pick = probability if pick == record["home_team"] else 1.0 - probability

    if edge < NO_READ_EDGE:
        signal = (
            f"no read — {probability:.1%} is within {NO_READ_EDGE:.2f} of a coin flip"
        )
    elif edge < CONFIDENCE_BANDS[2][0]:
        # D-21: the two middle bands could not be told apart on validation, so
        # they are presented as one undifferentiated read. The bands themselves
        # are untouched — the collapse is a narrative choice, not a refit.
        signal = (
            f"thin edge — {for_pick:.1%} for {pick}; validation could not "
            f"separate the two middle bands, so they are reported as one"
        )
    else:
        signal = f"clear edge — {for_pick:.1%} for {pick}"

    votes = record["model_votes"]
    agreed = sum(vote["pick"] == pick for vote in votes.values())
    spread = float(np.std([vote["home_win_prob"] for vote in votes.values()]))
    agreement = (
        f"{agreed} of {len(votes)} members pick {pick}; "
        f"member probabilities spread {spread:.3f}"
    )
    if agreed < len(votes):
        dissenters = [n for n, v in votes.items() if v["pick"] != pick]
        agreement += f" ({', '.join(dissenters)} dissenting)"
    else:
        agreement += " — unanimous, which is the sharper of the two drivers here"

    quality: list[str] = []
    week = int(row["week"])
    if week <= 4:
        quality.append(
            f"week {week}, so the rolling form windows still reach back into last season"
        )
    for side, team in (("home", record["home_team"]), ("away", record["away_team"])):
        if int(row[f"{side}_backup_qb_starting"]):
            quality.append(f"{team} is starting other than its usual quarterback")
        if int(row[f"{side}_qb_form_unknown"]):
            quality.append(f"no prior form for {team}'s projected quarterback")
    missing = [name for name in features if row[name] is None]
    if missing:
        quality.append(f"{len(missing)} of {len(features)} inputs missing")

    return {
        "signal_strength": signal,
        "model_agreement": agreement,
        "data_quality": (
            "full — no early-season, quarterback or missing-input caveats"
            if not quality
            else "reduced — " + "; ".join(quality)
        ),
    }


def dissent(
    record: Mapping[str, Any],
    attributions: Mapping[str, Attribution],
    position: int,
    row: Mapping[str, Any],
    features: Sequence[str] = ENSEMBLE_FEATURES,
) -> str:
    """Which members disagree, and what each one is weighting to get there.

    Expected to be thin. D-21 measured the six members correlating at a mean of
    0.959, so genuine per-model disagreement is rare — if every game produced a
    rich dissent paragraph, this function would be inventing a diversity the
    ensemble does not have.

    Votes come from the record, so this names exactly the dissenters the record
    shows. The *reason* comes from the member's own attribution, which is of the
    same fitted model before its sigmoid — monotone, so the feature ranking is
    unchanged even though the probability is not.
    """
    pick = record["ensemble"]["pick"]
    home, away = record["home_team"], record["away_team"]
    dissenters = [n for n, v in record["model_votes"].items() if v["pick"] != pick]
    if not dissenters:
        return f"none — all {len(record['model_votes'])} members pick {pick}."

    parts: list[str] = []
    for name in dissenters:
        other = record["model_votes"][name]["pick"]
        values = attributions[name].values[position]
        best = int(np.argmax(np.abs(values)))
        if abs(values[best]) < FACTOR_FLOOR:
            parts.append(f"{name} picks {other}, on no single dominant input")
            continue
        phrase = describe(features[best], float(row[features[best]]), home, away)
        parts.append(f"{name} picks {other}, weighting {phrase} most heavily")
    return "; ".join(parts) + "."


# ------------------------------------------------------------- the narrative

def narrative(
    probability: float,
    record: Mapping[str, Any],
    factors_for: Sequence[Mapping[str, Any]],
    factors_against: Sequence[Mapping[str, Any]],
    drivers: Mapping[str, str],
    baselines: Mapping[str, float | None],
) -> str:
    """Deterministic templates. No LLM, by requirement, not by preference.

    A stochastic generator would make it impossible to tell whether a changed
    explanation reflects a changed model or just sampling noise — which is the
    whole reason the spec forbids one.

    Every phrase here describes **model behaviour**: what the model weights,
    never what wins football games. :data:`BANNED_PHRASES` is the executable
    form of that constraint.
    """
    pick = record["ensemble"]["pick"]
    home, away = record["home_team"], record["away_team"]
    other = away if pick == home else home

    if abs(probability - 0.5) < NO_READ_EDGE:
        return (
            f"The model has no meaningful read on {away} at {home}: it lands at "
            f"{probability:.1%} for {home}, inside {NO_READ_EDGE:.2f} of a coin flip. "
            f"No ranked rationale is offered, since any factor list assembled at this "
            f"margin would describe rounding rather than a judgement. "
            f"For the record, {drivers['model_agreement']}."
        )

    votes = record["model_votes"]
    agreed = sum(v["pick"] == pick for v in votes.values())
    spread = float(np.std([v["home_win_prob"] for v in votes.values()]))

    for_pick = probability if pick == home else 1.0 - probability
    parts = [f"The model favours {pick} over {other}, at {for_pick:.1%} for {pick}."]
    if factors_for:
        parts.append(
            "It weights most heavily toward that side: "
            + "; ".join(f["plain"] for f in factors_for)
            + "."
        )
    if factors_against:
        # "Weighs against" describes the attribution, not the feature: a column
        # can favour the picked team on its face and still be weighted against
        # it by the model. That is the distinction SHAP exists to make, so the
        # sentence is worded to keep the two apart.
        parts.append(
            "It weighs these against that side: "
            + "; ".join(f["plain"] for f in factors_against)
            + "."
        )
    if not factors_for and not factors_against:
        parts.append("No single input clears the threshold to be named a factor.")

    edge_word = "clear" if abs(probability - 0.5) >= CONFIDENCE_BANDS[2][0] else "thin"
    parts.append(
        f"Confidence is {record['ensemble']['confidence']}: the edge is {edge_word}, and "
        f"{agreed} of {len(votes)} members agree (member probabilities spread {spread:.3f})."
    )

    elo, market = baselines.get("elo_only"), baselines.get("market_implied")
    anchors = [
        f"Elo alone {elo:.1%}" if elo is not None else None,
        f"the market {market:.1%}" if market is not None else None,
    ]
    parts.append("For reference: " + ", ".join(a for a in anchors if a) + ".")

    far = [
        anchor
        for anchor in (elo, market)
        if anchor is not None and abs(probability - anchor) > ANCHOR_GAP
    ]
    if len(far) == 2:
        parts.append(
            f"This read sits more than {ANCHOR_GAP:.0%} from both anchors, which is worth "
            f"treating as a flag on the model rather than as a discovery."
        )
    return " ".join(parts)


# ------------------------------------------------------------- the assembly

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
    :func:`explained_records` does the merge, which lands ``explanation`` last —
    exactly where the spec's example has it.

    ``probability`` is the ensemble's **unrounded** output. It is a parameter
    rather than a read of ``record["ensemble"]["home_win_prob"]`` because that
    field is rounded to four places for display, and reconciling against it
    would report a residual of ~6e-05 that measures the rounding rather than the
    attribution — burying the very thing the field exists to prove.
    """
    home, away = record["home_team"], record["away_team"]

    no_read = abs(probability - 0.5) < NO_READ_EDGE
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
                "approximation — per-model SHAP combined by the meta-learner's "
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
            "base_prob": round(float(_sigmoid(origin)), 4),
            "reconciliation": round(
                abs(total - (_logit(probability) - origin)), 9
            ),
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
    ``probability`` is the ensemble's unrounded output — see
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
