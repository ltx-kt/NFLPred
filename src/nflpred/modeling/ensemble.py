"""The Phase 5 ensemble layer: six members, one probability, and the plumbing.

Phase 4 found no winner among the five estimators — 0.0079 of log loss separates
all ten fitted models, and none of them clears the Phase 3 bar. Phase 5 is the
spec's answer: combine them, and say plainly whether that helps.

**The one design choice everything else follows from.** Elo is a member but not
an sklearn estimator — the rating system emits a probability directly, and
`elo_prob` is a column of the feature matrix. Rather than special-case it at
every call site, the matrix column rides along in the feature tuple:

    ENSEMBLE_FEATURES = (*CORE_FEATURES, "elo_prob")     # 16 columns, then one

Every ensemble object is then a plain ``predict_proba(X)`` estimator that slices
internally — the five sklearn members read ``X[:, :16]``, the Elo member reads
``X[:, 16]``. That is what lets
:func:`~nflpred.modeling.base.home_win_probability`, :func:`~nflpred.modeling.store.save_models`
and :func:`~nflpred.modeling.store.load_models` work on an ensemble **unchanged**: they
already take a ``(model, frame, features)`` triple and do
``frame.select(features).to_numpy()``. Passing `elo_diff` instead would not do —
`elo_prob` carries the home-field and neutral-site handling `nflpred.features.elo`
applies, and `elo_diff` alone would make the Elo member a different model.

**Why not FrozenEstimator here.** `nflpred.modeling.base` wraps a fitted base in
`FrozenEstimator` for calibration (D-13), so its absence here reads as an
oversight unless stated: `FrozenEstimator` **cannot be a VotingClassifier
member**. `VotingClassifier._validate_estimators` runs `is_classifier` on each
member, and a frozen wrapper does not answer to it — "The estimator
FrozenEstimator should be a classifier". :class:`PrefitMember` is the
project-owned replacement, and subclasses `ClassifierMixin` **first** in the MRO
because under the 1.6+ tags API putting `BaseEstimator` first silently fails
`is_classifier` rather than erroring.

**Why the OOF stack is hand-rolled.** The spec names
``StackingClassifier(..., cv=TimeSeriesSplit(...))``. That does not run on
scikit-learn 1.9: `StackingClassifier` builds its meta-features with
`cross_val_predict`, which raises ``"cross_val_predict only works for
partitions"`` — and a `TimeSeriesSplit` is not a partition, since the earliest
block is never in a test fold. :class:`TimeSeriesStack` does what the spec asked
for, forward-chained and honest about the rows it drops. See D-20.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

from nflpred.config import RANDOM_STATE
from nflpred.features.build import CORE_FEATURES, to_training_arrays
from nflpred.modeling.base import (
    build_models,
    elo_platt_probability,
    fit_weighted,
    home_win_probability,
)

#: The 16 modelling columns plus Elo's probability as column 16. Seventeen
#: columns so that nothing downstream of this module needs an edit.
ENSEMBLE_FEATURES: Final[tuple[str, ...]] = (*CORE_FEATURES, "elo_prob")

#: Column indices into a matrix built from :data:`ENSEMBLE_FEATURES`.
BASE_COLUMNS: Final[tuple[int, ...]] = tuple(range(len(CORE_FEATURES)))
ELO_COLUMN: Final[int] = len(CORE_FEATURES)

#: The name of the Elo member, and the fixed order of all six. Every table,
#: coefficient print and per-game record uses this order, so a column of numbers
#: means the same thing wherever it appears.
ELO_MEMBER: Final[str] = "elo"
MEMBER_ORDER: Final[tuple[str, ...]] = (
    "logreg",
    "random forest",
    "xgboost",
    "lightgbm",
    "catboost",
    ELO_MEMBER,
)

#: Confidence bands on |p - 0.50|, fixed **a priori** rather than fitted. Upper
#: bound of each band, exclusive, and the label it earns. The spec asks for
#: confidence derived from calibrated probability rather than vote count, and a
#: band chosen after seeing which cuts flatter the validation numbers would be a
#: selection artefact dressed as a scale. The checkpoint reports the hit rate per
#: bucket, which is how these get judged.
#:
#: 0.03 is the "no read" cut: a 47-53% pick is the model declining to have an
#: opinion, and saying so is more useful than a coin flip with a label on it.
CONFIDENCE_BANDS: Final[tuple[tuple[float, str], ...]] = (
    (0.03, "no read"),
    (0.07, "low"),
    (0.13, "moderate"),
    (float("inf"), "high"),
)


def confidence(probability: float) -> str:
    """The a priori confidence band for one calibrated probability."""
    edge = abs(float(probability) - 0.5)
    return next(label for cut, label in CONFIDENCE_BANDS if edge < cut)


# --------------------------------------------------------------- the members


class PrefitMember(ClassifierMixin, BaseEstimator):
    """One already-fitted estimator, presented to sklearn as an unfitted member.

    ``fit`` is a no-op and ``__sklearn_clone__`` returns ``self``, so neither
    `VotingClassifier` nor `StackingClassifier` can refit the wrapped model no
    matter which of them holds it — the members stay exactly the objects Phase 4
    fitted on 2006-2015 and calibrated on 2016-2018. `tests/test_ensemble.py`
    pins that by comparing every member's validation predictions before and
    after each ensemble fit.

    ``columns`` is the slice of the 17-column ensemble matrix this member was
    fitted on. See the module docstring for why `FrozenEstimator` is not used.
    """

    def __init__(
        self, estimator: ClassifierMixin | None = None, columns: Sequence[int] | None = None
    ) -> None:
        self.estimator = estimator
        self.columns = columns

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None, sample_weight: np.ndarray | None = None
    ) -> PrefitMember:
        """Learn nothing. The wrapped estimator is already fitted."""
        self.classes_ = np.array([0.0, 1.0])
        return self

    def __sklearn_clone__(self) -> PrefitMember:
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return True

    def _slice(self, X: np.ndarray) -> np.ndarray:
        columns = BASE_COLUMNS if self.columns is None else self.columns
        return np.asarray(X, dtype=float)[:, list(columns)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(self._slice(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(float)


class ColumnSubset(ClassifierMixin, BaseEstimator):
    """The refitting twin of :class:`PrefitMember`, for the spec-literal stack.

    Normal clone semantics and a real ``fit``: :class:`TimeSeriesStack` needs
    members it *can* refit fold by fold, on their own 16 columns, out of the
    17-column matrix. Weights reach the wrapped estimator through
    :func:`~nflpred.modeling.base.fit_weighted`, so a `Pipeline` member is routed the
    same way here as everywhere else (D-4).
    """

    def __init__(
        self, estimator: ClassifierMixin | None = None, columns: Sequence[int] | None = None
    ) -> None:
        self.estimator = estimator
        self.columns = columns

    def _slice(self, X: np.ndarray) -> np.ndarray:
        columns = BASE_COLUMNS if self.columns is None else self.columns
        return np.asarray(X, dtype=float)[:, list(columns)]

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> ColumnSubset:
        self.estimator_ = clone(self.estimator)
        x = self._slice(X)
        if sample_weight is None:
            self.estimator_.fit(x, y)
        else:
            fit_weighted(self.estimator_, x, y, sample_weight)
        self.classes_ = np.asarray(self.estimator_.classes_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator_.predict_proba(self._slice(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(float)


class EloMember(ClassifierMixin, BaseEstimator):
    """Elo as an ensemble member: read column 16, optionally Platt-scale it.

    Nothing is fitted here in either mode. With ``platt=None`` this is exactly
    :func:`~nflpred.modeling.base.elo_probability` — a test pins that equality — which
    is also why it is safe inside :class:`TimeSeriesStack`: a member with no
    parameters cannot leak across a fold boundary. With a ``platt`` fitted on the
    calibration seasons it is the "elo only (calibrated)" comparator, so the
    ensemble's Elo member and the checkpoint's Elo row are the same object.
    """

    def __init__(
        self, platt: LogisticRegression | None = None, column: int | None = None
    ) -> None:
        self.platt = platt
        self.column = column

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None, sample_weight: np.ndarray | None = None
    ) -> EloMember:
        """Learn nothing. The rating system already emitted a probability."""
        self.classes_ = np.array([0.0, 1.0])
        return self

    def __sklearn_clone__(self) -> EloMember:
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        column = ELO_COLUMN if self.column is None else self.column
        probability = np.asarray(X, dtype=float)[:, column]
        if self.platt is not None:
            probability = elo_platt_probability(self.platt, probability)
        return np.column_stack([1.0 - probability, probability])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(float)


Member = tuple[str, ClassifierMixin]


def build_members(
    estimators: Mapping[str, ClassifierMixin],
    platt: LogisticRegression | None = None,
) -> list[Member]:
    """The six members, in :data:`MEMBER_ORDER`, ready for either meta-estimator.

    ``estimators`` supplies the five sklearn models under their project names —
    calibrated wrappers or the bare fitted bases, depending on which variant is
    being built. Elo is appended rather than looked up: it is not in the dict and
    never will be.
    """
    members: list[Member] = [
        (name, PrefitMember(estimators[name], BASE_COLUMNS))
        for name in MEMBER_ORDER
        if name != ELO_MEMBER
    ]
    members.append((ELO_MEMBER, EloMember(platt=platt)))
    return members


# ------------------------------------------------------------ the ensembles


def _meta_learner() -> LogisticRegression:
    """The stacking meta-learner. Linear on purpose — the spec's SHAP-combination
    step needs coefficients it can use as weights, and says to skip that step
    entirely if the meta-learner is nonlinear."""
    return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)


def soft_vote(
    members: Sequence[Member],
    calib: pl.DataFrame,
    features: Sequence[str] = ENSEMBLE_FEATURES,
    weights: Sequence[float] | None = None,
    weight: str | None = None,
) -> VotingClassifier:
    """An equal-weight (or ``weights``-weighted) average of the members.

    Returned **fitted**, because sklearn requires a fit before ``predict_proba``.
    That fit learns nothing: every member's ``fit`` is a no-op, so all the call
    establishes is the label encoder. It runs on the calibration split so that
    all four ensemble objects are built from the same rows and the checkpoint
    table is a like-for-like comparison.

    Two weight arguments, and they are not the same thing. ``weights`` is
    sklearn's per-**member** vote weighting; ``weight`` names a per-**row**
    column, the Phase 7 recency channel. It changes nothing here — this fit
    learns nothing to weight — and is accepted so that every ensemble builder
    takes the same argument and a caller cannot pass it to three of four.
    """
    vote = VotingClassifier(list(members), voting="soft", weights=weights)
    x, y, w = to_training_arrays(calib, features, weight=weight)
    vote.fit(x, y, sample_weight=w)
    return vote


def prefit_stack(
    members: Sequence[Member],
    calib: pl.DataFrame,
    features: Sequence[str] = ENSEMBLE_FEATURES,
    weight: str | None = None,
) -> StackingClassifier:
    """Members held fixed, meta-learner fitted on the calibration split.

    ``cv="prefit"`` is the D-13 counterpart that survived: unlike
    ``CalibratedClassifierCV(cv='prefit')``, which was removed in scikit-learn
    1.9, this one exists and is not deprecated. It does not refit the members,
    and ``fit(X, y, sample_weight=w)`` forwards the weight to the final
    estimator — so D-4's tie encoding reaches the meta-learner, and with
    ``weight`` set so does Phase 7's recency decay.
    """
    stack = StackingClassifier(list(members), final_estimator=_meta_learner(), cv="prefit")
    x, y, w = to_training_arrays(calib, features, weight=weight)
    stack.fit(x, y, sample_weight=w)
    return stack


class TimeSeriesStack(ClassifierMixin, BaseEstimator):
    """Forward-chained out-of-fold stacking — the spec's variant, hand-rolled.

    `StackingClassifier` cannot take a `TimeSeriesSplit` (see the module
    docstring), so the meta-feature matrix is built here: for each forward-
    chained fold the members are refitted on the earlier rows and asked to
    predict the later ones, and the meta-learner is fitted on the rows that were
    covered. The earliest block is covered by no fold and is dropped, which is
    the price of not training a member on rows it will then be scored on;
    ``n_oof_`` records how many rows survived so the cost is a number rather
    than an assumption.

    Members are then refitted once on the whole frame for inference, which is
    what `StackingClassifier` does too.

    Unfitted on construction, so :func:`~nflpred.modeling.base.fit_calibrated` fits it
    on 2006-2015 and sigmoid-calibrates it on 2016-2018 like any other
    estimator — no second fit path exists for the ensemble.
    """

    def __init__(
        self,
        estimators: Sequence[Member] | None = None,
        final_estimator: ClassifierMixin | None = None,
        n_splits: int = 5,
    ) -> None:
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.n_splits = n_splits

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> TimeSeriesStack:
        x = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        meta = np.full((len(y), len(self.estimators)), np.nan)
        for train_index, test_index in TimeSeriesSplit(n_splits=self.n_splits).split(x):
            fold_weight = None if sample_weight is None else sample_weight[train_index]
            for position, (_, estimator) in enumerate(self.estimators):
                fold = clone(estimator)
                fold.fit(x[train_index], y[train_index], sample_weight=fold_weight)
                meta[test_index, position] = fold.predict_proba(x[test_index])[:, 1]

        covered = ~np.isnan(meta).any(axis=1)
        self.n_oof_ = int(covered.sum())

        self.final_estimator_ = clone(self.final_estimator)
        self.final_estimator_.fit(
            meta[covered],
            y[covered],
            sample_weight=None if sample_weight is None else sample_weight[covered],
        )

        self.estimators_ = []
        for _, estimator in self.estimators:
            full = clone(estimator)
            full.fit(x, y, sample_weight=sample_weight)
            self.estimators_.append(full)

        self.classes_ = np.array([0.0, 1.0])
        return self

    def _meta_features(self, X: np.ndarray) -> np.ndarray:
        x = np.asarray(X, dtype=float)
        return np.column_stack([e.predict_proba(x)[:, 1] for e in self.estimators_])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.final_estimator_.predict_proba(self._meta_features(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(float)


def oof_stack(
    features: Sequence[str] = CORE_FEATURES, n_splits: int = 5
) -> TimeSeriesStack:
    """An **unfitted** spec-literal stack, for :func:`fit_calibrated` to fit.

    ``features`` selects the members' frozen hyperparameters (D-16), not their
    columns — the members always read the first 16 columns of the ensemble
    matrix. Elo joins as a parameter-free :class:`EloMember`, which needs no
    refitting-twin because there is nothing in it to refit.
    """
    members: list[Member] = [
        (name, ColumnSubset(estimator, BASE_COLUMNS))
        for name, estimator in build_models(features).items()
    ]
    members.append((ELO_MEMBER, EloMember()))
    return TimeSeriesStack(members, final_estimator=_meta_learner(), n_splits=n_splits)


# --------------------------------------------------------------- the records


def meta_coefficients(stack: StackingClassifier | TimeSeriesStack) -> dict[str, float]:
    """The meta-learner's weight per member, in :data:`MEMBER_ORDER`.

    Phase 6 treats these as authoritative for combining per-model SHAP values,
    which is what closes the spec's "ensemble weight ownership" open question —
    so they are printed at the checkpoint rather than left inside a pickle.
    """
    names = [name for name, _ in stack.estimators]
    coefficients = np.asarray(stack.final_estimator_.coef_).ravel()
    return dict(zip(names, (float(c) for c in coefficients), strict=True))


def member_probabilities(
    members: Sequence[Member], frame: pl.DataFrame, features: Sequence[str] = ENSEMBLE_FEATURES
) -> dict[str, np.ndarray]:
    """Each member's P(home win) on ``frame``, one array per member.

    Not routed through :func:`~nflpred.features.build.to_training_arrays`: ties are
    expanded for fitting, but a tie game still gets exactly one prediction.
    """
    x = frame.select(features).to_numpy().astype(float)
    return {name: member.predict_proba(x)[:, 1] for name, member in members}


def predict_records(
    ensemble: ClassifierMixin,
    members: Sequence[Member],
    frame: pl.DataFrame,
    features: Sequence[str] = ENSEMBLE_FEATURES,
) -> list[dict[str, Any]]:
    """The spec's per-game record, minus the ``explanation`` key Phase 6 adds.

    ``agreement`` counts members whose *pick* matches the ensemble's, which is
    the spec's "5 of 6 models agree" and deliberately not a probability spread —
    the spread is reported separately as the correlation diagnostic. A game at
    exactly 0.500 is a home pick, matching the threshold
    :func:`~nflpred.evaluate.accuracy` scores on.
    """
    ensemble_probability = home_win_probability(ensemble, frame, features)
    per_member = member_probabilities(members, frame, features)

    def pick(probability: float, home: str, away: str) -> str:
        return home if probability >= 0.5 else away

    records: list[dict[str, Any]] = []
    for row, (game_id, home, away) in enumerate(
        zip(frame["game_id"], frame["home_team"], frame["away_team"], strict=True)
    ):
        votes = {
            name: {
                "pick": pick(per_member[name][row], home, away),
                "home_win_prob": round(float(per_member[name][row]), 4),
            }
            for name, _ in members
        }
        ensemble_pick = pick(ensemble_probability[row], home, away)
        agreed = sum(vote["pick"] == ensemble_pick for vote in votes.values())

        records.append(
            {
                "game_id": game_id,
                "home_team": home,
                "away_team": away,
                "model_votes": votes,
                "ensemble": {
                    "pick": ensemble_pick,
                    "home_win_prob": round(float(ensemble_probability[row]), 4),
                    "confidence": confidence(ensemble_probability[row]),
                },
                "agreement": f"{agreed}/{len(members)}",
            }
        )
    return records
