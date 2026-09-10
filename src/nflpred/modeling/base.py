"""The five base estimators, and the one way they are fitted and calibrated.

This module is the factory and the source of truth for what a Phase 4 model
*is*. Artifacts on disk are outputs of it (D-17), never a substitute for it: if
these definitions and a pickle disagree, the definitions win.

**How calibration works here.** D-5 requires the base estimator to be fitted on
2006-2015 and the calibrator on 2016-2018, with the two never mixed. The spec
names ``CalibratedClassifierCV(cv='prefit')`` for that, which no longer exists —
it was removed in scikit-learn 1.9. The replacement is
``CalibratedClassifierCV(FrozenEstimator(fitted_base))``, which preserves the
property D-5 actually depended on: exactly **one** inner estimator, inspectable
by Phase 6's SHAP layer, rather than the *k* that internal cross-validation
would leave behind. See D-13.

**How ties get in.** Everything goes through
:func:`~nflpred.features.build.to_training_arrays`, so the D-4 two-row tie encoding
rides the ``sample_weight`` channel into both the base fit and the calibrator
fit. :func:`fit_weighted` is the single place that knows how to hand a weight
to an estimator, which is what makes that claim checkable.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

import numpy as np
import polars as pl
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from nflpred.config import MODEL_PARAM_OVERRIDES, MODEL_PARAMS, N_JOBS, RANDOM_STATE
from nflpred.features.build import CORE_FEATURES, to_training_arrays

#: Calibrating a frozen estimator with sample weights warns that the weights
#: reached only the calibrator. That is precisely the intent — the base is
#: already fitted, and D-4's tie weights are being applied to the sigmoid — so
#: the warning is expected and suppressed narrowly, by message, at the one call
#: site that provokes it.
_FROZEN_WEIGHT_WARNING: Final[str] = "Since FrozenEstimator does not appear to accept sample_weight"


def _logreg(params: Mapping[str, Any]) -> Pipeline:
    """Scaled L2 logistic regression — the Phase 2/3 estimator, now tuned."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, **params)),
        ]
    )


def _random_forest(params: Mapping[str, Any]) -> RandomForestClassifier:
    """Random forest — the one estimator pinned to a single thread.

    ``min_samples_leaf`` is gridded well above sklearn's default of 1: on 2,670
    games the default grows leaves down to single games, which memorises the
    training seasons and predicts noise.

    ``n_jobs=1`` is not a performance choice. sklearn's forest sums each tree's
    contribution into a shared output buffer from worker threads, so the order
    of that sum is not fixed — measured here, a forest at ``n_jobs=4`` gives
    answers that differ in the last bit *between two calls on the same fitted
    model*. That is a millionth of nothing for a prediction and fatal for the
    D-17 fingerprint, which exists to distinguish "this is the model that was
    saved" from "this is something else". One thread costs ~0.5s per fit and
    buys an exactly reproducible model. The three boosters do not need it.
    """
    return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, **params)


def _xgboost(params: Mapping[str, Any]) -> XGBClassifier:
    """XGBoost. ``n_jobs`` is pinned — thread count changes float summation order."""
    return XGBClassifier(
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        tree_method="hist",
        eval_metric="logloss",
        **params,
    )


def _lightgbm(params: Mapping[str, Any]) -> LGBMClassifier:
    """LightGBM.

    ``verbosity=-1`` because on a set this small it otherwise warns about splits
    with no positive gain on nearly every fit; ``deterministic`` plus
    ``force_row_wise`` because reproducibility is a Phase 4 test, not a hope.
    """
    return LGBMClassifier(
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbosity=-1,
        deterministic=True,
        force_row_wise=True,
        **params,
    )


def _catboost(params: Mapping[str, Any]) -> CatBoostClassifier:
    """CatBoost. ``allow_writing_files=False`` or it litters ``catboost_info/``."""
    return CatBoostClassifier(
        random_seed=RANDOM_STATE,
        thread_count=N_JOBS,
        verbose=0,
        allow_writing_files=False,
        **params,
    )


#: The five estimators, by the name they carry through every table, artifact and
#: manifest in the project. A bare `DecisionTreeClassifier` is deliberately
#: absent — the spec says it is dominated by the forest and only adds noise.
BUILDERS: Final[dict[str, Callable[[Mapping[str, Any]], ClassifierMixin]]] = {
    "logreg": _logreg,
    "random forest": _random_forest,
    "xgboost": _xgboost,
    "lightgbm": _lightgbm,
    "catboost": _catboost,
}


def params_for(name: str, feature_set: Sequence[str]) -> dict[str, Any]:
    """The frozen hyperparameters for one estimator on one feature set (D-16).

    Overrides are keyed by feature count, which is how the grid reported them.
    A set with no entry gets the shared constants — true of `wide_features()`
    at 104 columns, which is a control rather than a candidate and was never
    gridded.
    """
    params = dict(MODEL_PARAMS.get(name, {}))
    params.update(MODEL_PARAM_OVERRIDES.get(len(feature_set), {}).get(name, {}))
    return params


def build_models(feature_set: Sequence[str] = CORE_FEATURES) -> dict[str, ClassifierMixin]:
    """The five unfitted estimators, with their frozen hyperparameters.

    ``feature_set`` selects the parameters rather than the columns: two of the
    five estimators were gridded to different settings on 16 columns and on 29,
    and using the 16-column settings for both would tilt the feature-count
    comparison the Phase 4 checkpoint exists to settle.
    """
    return {name: builder(params_for(name, feature_set)) for name, builder in BUILDERS.items()}


def fit_weighted(
    estimator: ClassifierMixin, x: np.ndarray, y: np.ndarray, weight: np.ndarray
) -> ClassifierMixin:
    """Fit, handing ``weight`` to whatever actually accepts it.

    The one place that knows a `Pipeline` routes sample weights by step name
    while a bare estimator takes them directly. D-4's tie weights are useless if
    they are silently dropped here, so there is exactly one such place and
    `tests/test_models.py` asserts every estimator honours it.
    """
    if isinstance(estimator, Pipeline):
        estimator.fit(x, y, **{f"{estimator.steps[-1][0]}__sample_weight": weight})
    else:
        estimator.fit(x, y, sample_weight=weight)
    return estimator


def fit_calibrated(
    estimator: ClassifierMixin,
    train: pl.DataFrame,
    calib: pl.DataFrame,
    features: Sequence[str],
    method: str = "sigmoid",
    weight: str | None = None,
) -> CalibratedClassifierCV:
    """Fit ``estimator`` on ``train``, then a calibrator on ``calib`` (D-5, D-13).

    ``method`` is ``"sigmoid"`` everywhere by default. Isotonic is a diagnostic
    row rather than the standard: 801 calibration games is thin for a
    non-parametric fit, and a calibrator that overfits its own split is worse
    than none.

    ``weight`` names a column of both frames to multiply into the sample
    weights, which is how Phase 7's recency decay reaches a fit (D-26). It is
    forwarded to **both** calls deliberately: the calibrator is the half D-18
    found had been fitted to an era that ended, so weighting the base fit and
    leaving the sigmoid flat would be fixing the smaller of the two problems.
    """
    x_train, y_train, w_train = to_training_arrays(train, features, weight=weight)
    fit_weighted(estimator, x_train, y_train, w_train)

    x_calib, y_calib, w_calib = to_training_arrays(calib, features, weight=weight)
    calibrated = CalibratedClassifierCV(FrozenEstimator(estimator), method=method)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=re.escape(_FROZEN_WEIGHT_WARNING))
        calibrated.fit(x_calib, y_calib, sample_weight=w_calib)
    return calibrated


def fit_all(
    matrix: pl.DataFrame,
    features: Sequence[str] = CORE_FEATURES,
    method: str = "sigmoid",
    weight: str | None = None,
) -> dict[str, CalibratedClassifierCV]:
    """Fit and calibrate all five estimators from one feature matrix.

    Reads only the ``train`` and ``calib`` splits. Validation is scored, never
    fitted; test is not read at all, and `tests/test_models.py` pins that.

    The frozen-split convenience wrapper. Phase 7's walk-forward harness calls
    :func:`fit_calibrated` directly instead, because the two windows it needs
    move week by week and cannot be named by :func:`split_frame` — which is the
    seam D-26 turns on.
    """
    train, calib = split_frame(matrix, "train"), split_frame(matrix, "calib")
    return {
        name: fit_calibrated(estimator, train, calib, features, method=method, weight=weight)
        for name, estimator in build_models(features).items()
    }


def split_frame(matrix: pl.DataFrame, name: str) -> pl.DataFrame:
    """One split, in kickoff order. Sorting is not cosmetic: `TimeSeriesSplit`
    inside the training seasons and the D-17 fingerprint both assume it."""
    return matrix.filter(pl.col("split") == name).sort("gameday", "game_id")


def home_win_probability(
    model: CalibratedClassifierCV, frame: pl.DataFrame, features: Sequence[str]
) -> np.ndarray:
    """Calibrated P(home win), one per row of ``frame``, in ``frame``'s order.

    Deliberately not routed through :func:`~nflpred.features.build.to_training_arrays`:
    ties are expanded for *fitting*, but a tie game still gets exactly one
    prediction.
    """
    return model.predict_proba(frame.select(features).to_numpy().astype(float))[:, 1]


def elo_probability(frame: pl.DataFrame) -> np.ndarray:
    """The Elo-only baseline. No estimator — the rating system already emits a
    probability, and `elo_prob` was computed before kickoff like every other
    column in the matrix."""
    return frame["elo_prob"].to_numpy().astype(float)


def elo_logit(probability: np.ndarray) -> np.ndarray:
    """Log-odds of an Elo probability, clipped so no rating gap yields ±inf."""
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_elo_platt(calib: pl.DataFrame, weight: str | None = None) -> LogisticRegression:
    """Platt-scale Elo on the calibration seasons, and return the scaler.

    Elo has no base fit to freeze — the rating system already emits a
    probability — so its calibrator is a plain logistic regression on the logit,
    fitted on 2016-2018 and nothing else. That gives it exactly the calibration
    split the five estimators get (D-5), and makes "is Elo's raw probability
    already well calibrated?" a number rather than an impression.

    Returns the fitted estimator rather than an array of probabilities because
    three phases now need it: Phase 4 scores a row with it, Phase 5 holds it
    inside :class:`~nflpred.modeling.ensemble.EloMember`, and Phase 7 refits it on a
    rolling calibration window every week.

    ``weight`` takes the same recency column the five estimators take, so Elo's
    calibrator is not the one member left fitted flat across its window.
    """
    frame = calib.with_columns(elo_logit=pl.Series(elo_logit(elo_probability(calib))))
    x, y, w = to_training_arrays(frame, ["elo_logit"], weight=weight)
    return LogisticRegression(max_iter=1000).fit(x, y, sample_weight=w)


def elo_platt_probability(platt: LogisticRegression, probability: np.ndarray) -> np.ndarray:
    """Apply a fitted Elo scaler to raw Elo probabilities."""
    return platt.predict_proba(elo_logit(probability).reshape(-1, 1))[:, 1]


def inner_estimator(model: CalibratedClassifierCV) -> ClassifierMixin:
    """The fitted base estimator inside a calibrated wrapper.

    The one place in the project that knows this path. Phase 6 explains the
    *uncalibrated* model — `TreeExplainer` cannot see through a sigmoid — so it
    needs this object, and it should not need to know that
    ``calibrated_classifiers_[0].estimator`` is a `FrozenEstimator` whose own
    ``.estimator`` is the thing that was fitted. If sklearn moves it again, this
    function is the only edit.
    """
    if len(model.calibrated_classifiers_) != 1:
        msg = (
            f"expected exactly one inner estimator (D-5), found "
            f"{len(model.calibrated_classifiers_)}. The calibrator was fitted with "
            f"cross-validation instead of on a frozen base."
        )
        raise ValueError(msg)

    inner = model.calibrated_classifiers_[0].estimator
    return inner.estimator if isinstance(inner, FrozenEstimator) else inner
