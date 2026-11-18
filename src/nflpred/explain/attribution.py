"""Per-member SHAP attributions and their combination through the meta-learner.

See the package docstring (D-25) for why the combination runs through secant
slopes rather than averaging raw SHAP values.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline

from nflpred.modeling.ensemble import (
    ELO_MEMBER,
    ENSEMBLE_FEATURES,
    MEMBER_ORDER,
    meta_coefficients,
)

#: Attribution spaces a member can live in.
MARGIN: Final[str] = "margin"
PROBABILITY: Final[str] = "probability"

#: Below this the secant slope is undefined - and the member's probability has
#: not moved either, so its contribution is zero rather than a divide-by-zero.
_FLAT: Final[float] = 1e-12


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
        """``(p_after, p_base)`` - where this member's output sits, and its origin."""
        if self.space == MARGIN:
            return sigmoid(self.base + self.totals()), float(sigmoid(self.base))
        return self.base + self.totals(), float(self.base)


@dataclass(frozen=True)
class Explainer:
    """How one member gets attributed. Built once per run; that is the slow part."""

    name: str
    estimator: ClassifierMixin | None
    shap: Any | None
    space: str
    base: float


def sigmoid(z: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def reference(name: str, estimator: ClassifierMixin, x: np.ndarray) -> np.ndarray:
    """The raw output an attribution must sum to, per member, in its own space.

    This is the additivity target, and it is deliberately each library's own
    uncalibrated call rather than anything routed through
    `CalibratedClassifierCV` - a member that fails to reconcile against this is
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
    rows - a per-row base means the explainer is doing something this module
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
        f"tolerance {atol:.0e}). The explainer is pointed at the wrong object - "
        f"most likely the calibrated wrapper rather than inner_estimator()."
    )
    raise ValueError(msg)


def build_explainers(
    estimators: Mapping[str, ClassifierMixin], elo_base: float
) -> dict[str, Explainer]:
    """One explainer per member. Construction is the expensive part, not use.

    ``estimators`` are the **raw** inner estimators - the objects the headline
    stack's meta-learner actually consumed - keyed by project name. ``elo_base``
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
            # value - the same kind of origin the tree explainers report.
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


def attribute_all(explainers: Mapping[str, Explainer], x: np.ndarray) -> dict[str, Attribution]:
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
