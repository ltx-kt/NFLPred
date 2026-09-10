"""The Phase 5 ensemble layer, and the claims that would fail silently.

`tests/test_models.py` covers the fit path; this file covers what is new at
Phase 5, which is almost entirely about things that *look* right when they are
wrong:

**Members that get refitted.** The whole design rests on `PrefitMember.fit`
being a no-op and `__sklearn_clone__` returning ``self``. If either broke, both
meta-estimators would refit the members on the calibration split — quietly, with
no error, producing an ensemble whose bases saw 2016-2018 and whose numbers
would look slightly *better*. So every member's validation predictions are
compared before and after each ensemble fit.

**A soft vote that is not a mean.** `VotingClassifier` weights, member order and
column slicing all have to line up; a wrong slice would produce a plausible
probability from the wrong columns.

**Weights that stop at the meta-learner.** D-4 rides `sample_weight` into
`StackingClassifier.fit`, and sklearn is free to drop it.

Runs against the real feature matrix, like `tests/test_models.py` — a synthetic
frame would not exercise the splits, the ties or the 17-column layout.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nflpred.config import FEATURE_MATRIX_GT_PATH
from nflpred.features.build import CORE_FEATURES, to_training_arrays
from nflpred.modeling.base import (
    elo_probability,
    fit_all,
    fit_calibrated,
    fit_elo_platt,
    home_win_probability,
    inner_estimator,
    split_frame,
)
from nflpred.modeling.ensemble import (
    BASE_COLUMNS,
    CONFIDENCE_BANDS,
    ELO_COLUMN,
    ENSEMBLE_FEATURES,
    MEMBER_ORDER,
    EloMember,
    PrefitMember,
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

PHASE5_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "phases" / "phase5_ensemble.py"


@pytest.fixture(scope="module")
def matrix() -> pl.DataFrame:
    if not FEATURE_MATRIX_GT_PATH.exists():
        pytest.skip(
            "game_features_gt.parquet not built; "
            "run python -m nflpred.features.build --garbage-time"
        )
    return pl.read_parquet(FEATURE_MATRIX_GT_PATH)


@pytest.fixture(scope="module")
def splits(matrix: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    return tuple(split_frame(matrix, name) for name in ("train", "calib", "val"))


@pytest.fixture(scope="module")
def models(matrix: pl.DataFrame) -> dict:
    return fit_all(matrix, CORE_FEATURES)


@pytest.fixture(scope="module")
def members(models: dict, splits) -> list:
    _, calib, _ = splits
    return build_members(models, platt=fit_elo_platt(calib))


# ------------------------------------------------------- the shape of a member


def test_ensemble_features_is_core_plus_one():
    """17 columns: the 16 that were fitted, then Elo's probability."""
    assert ENSEMBLE_FEATURES[: len(CORE_FEATURES)] == CORE_FEATURES
    assert ENSEMBLE_FEATURES[ELO_COLUMN] == "elo_prob"
    assert len(ENSEMBLE_FEATURES) == len(CORE_FEATURES) + 1
    assert list(BASE_COLUMNS) == list(range(len(CORE_FEATURES)))


def test_members_are_the_six_in_a_fixed_order(members):
    assert [name for name, _ in members] == list(MEMBER_ORDER)
    assert len(members) == 6


def test_elo_member_agrees_with_elo_probability(splits):
    """The raw Elo member is `elo_probability`, read out of column 16.

    Elo is the one member that is a matrix column rather than a fitted
    estimator, so "is it still the same baseline?" has to be an assertion.
    """
    _, _, val = splits
    x = val.select(ENSEMBLE_FEATURES).to_numpy().astype(float)

    np.testing.assert_array_equal(
        EloMember().predict_proba(x)[:, 1], elo_probability(val)
    )


def test_prefit_member_slices_the_first_sixteen_columns(models, splits):
    """A member's probability is its own model's, on its own columns."""
    _, _, val = splits
    member = PrefitMember(models["logreg"], BASE_COLUMNS)
    x = val.select(ENSEMBLE_FEATURES).to_numpy().astype(float)

    np.testing.assert_array_equal(
        member.predict_proba(x)[:, 1], home_win_probability(models["logreg"], val, CORE_FEATURES)
    )


# ------------------------------------------------- the whole safety argument


def test_no_ensemble_fit_touches_its_members(members, splits):
    """Members predict identically before and after every ensemble fit.

    The entire claim of the ``cv="prefit"`` / no-op-``fit`` design. If a member
    were refitted on the calibration split the ensemble would still run, still
    score, and be quietly contaminated.
    """
    _, calib, val = splits
    before = member_probabilities(members, val)

    soft_vote(members, calib)
    prefit_stack(members, calib)

    after = member_probabilities(members, val)
    for name in before:
        np.testing.assert_array_equal(
            before[name], after[name], err_msg=f"{name} was refitted by an ensemble fit"
        )


def test_soft_vote_is_the_mean_of_its_members(members, splits):
    _, calib, val = splits
    vote = soft_vote(members, calib)

    per_member = member_probabilities(members, val)
    expected = np.mean([per_member[name] for name, _ in members], axis=0)

    np.testing.assert_allclose(
        home_win_probability(vote, val, ENSEMBLE_FEATURES), expected, rtol=1e-12
    )


def test_a_one_member_vote_is_that_member(members, splits):
    _, calib, val = splits
    vote = soft_vote(members[:1], calib)

    np.testing.assert_allclose(
        home_win_probability(vote, val, ENSEMBLE_FEATURES),
        member_probabilities(members[:1], val)[members[0][0]],
        rtol=1e-12,
    )


def test_dropping_elo_changes_the_vote(members, splits):
    """The control row is a real control: the five-member vote is a different
    model, not a relabelled one."""
    _, calib, val = splits
    six = home_win_probability(soft_vote(members, calib), val, ENSEMBLE_FEATURES)
    five = home_win_probability(soft_vote(members[:-1], calib), val, ENSEMBLE_FEATURES)
    assert not np.allclose(six, five)


# ------------------------------------------------------ D-4 through the stack


def test_tie_weights_reach_the_meta_learner(members, splits):
    """Perturbing the sample weights moves the meta-learner's coefficients.

    `StackingClassifier.fit` forwards ``sample_weight`` to the final estimator.
    "It accepted the argument" is not the claim worth testing; "the argument had
    an effect" is, because a dropped weight would count every tie as two games.
    """
    _, calib, _ = splits

    even = prefit_stack(members, calib)
    baseline = np.array(list(meta_coefficients(even).values()))

    x, y, w = to_training_arrays(calib, ENSEMBLE_FEATURES)
    lopsided = w.copy()
    lopsided[len(lopsided) // 2 :] *= 0.001

    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression

    skewed = StackingClassifier(
        list(members), final_estimator=LogisticRegression(max_iter=1000), cv="prefit"
    )
    skewed.fit(x, y, sample_weight=lopsided)

    assert not np.allclose(baseline, np.asarray(skewed.final_estimator_.coef_).ravel()), (
        "the meta-learner's coefficients did not move under a 1000x weight "
        "imbalance — sample_weight is being dropped, and D-4's tie encoding "
        "never reaches the stack."
    )


def test_meta_coefficients_are_named_in_member_order(members, splits):
    _, calib, _ = splits
    coefficients = meta_coefficients(prefit_stack(members, calib))
    assert list(coefficients) == list(MEMBER_ORDER)
    assert all(isinstance(v, float) for v in coefficients.values())


def test_oof_stack_drops_only_the_earliest_block(matrix, splits):
    """The spec-literal stack keeps every row a forward-chained split can cover.

    `StackingClassifier` cannot take a `TimeSeriesSplit` at all (see D-20), so
    the cost of doing it properly is a number here rather than an assumption:
    five folds over 2,670 games leave the first sixth uncovered.
    """
    train, calib, _ = splits
    stack = fit_calibrated(oof_stack(CORE_FEATURES), train, calib, ENSEMBLE_FEATURES)
    inner = inner_estimator(stack)

    assert 0 < inner.n_oof_ < train.height
    assert inner.n_oof_ == pytest.approx(train.height * 5 / 6, abs=6)
    assert len(inner.estimators_) == len(MEMBER_ORDER)


# ------------------------------------------------------------ split hygiene


def test_phase5_script_never_names_the_unseen_split():
    """Static half: the checkpoint script cannot ask for held-out rows by name."""
    tree = ast.parse(PHASE5_SCRIPT.read_text(encoding="utf-8"))

    constants = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "test" not in constants

    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "TEST_SEASONS" not in imported


def test_ensembles_ignore_the_held_out_split(matrix, splits):
    """Runtime half: poison every held-out row and nothing moves.

    Stronger than reading the code — it would catch a join, a global mean, or a
    scaler that pulled the whole matrix in without asking for a split.
    """
    _, _, val = splits
    poisoned = matrix.with_columns(
        [
            pl.when(pl.col("split") == "test").then(999.0).otherwise(pl.col(c)).alias(c)
            for c in ENSEMBLE_FEATURES
        ]
        + [
            pl.when(pl.col("split") == "test")
            .then(1.0)
            .otherwise(pl.col("home_win"))
            .alias("home_win")
        ]
    )
    assert poisoned.filter(pl.col("split") == "test").height, "matrix must contain held-out rows"

    clean = _headline_probabilities(matrix, val)
    dirty = _headline_probabilities(poisoned, val)

    np.testing.assert_array_equal(
        clean, dirty, err_msg="the ensemble changed when the held-out split was rewritten"
    )


def _headline_probabilities(source: pl.DataFrame, val: pl.DataFrame) -> np.ndarray:
    train, calib = split_frame(source, "train"), split_frame(source, "calib")
    models = fit_all(source, CORE_FEATURES)
    members = build_members(models, platt=fit_elo_platt(calib))
    stack = prefit_stack(members, calib)
    vote = soft_vote(members, calib)
    return np.concatenate(
        [
            home_win_probability(stack, val, ENSEMBLE_FEATURES),
            home_win_probability(vote, val, ENSEMBLE_FEATURES),
            home_win_probability(
                fit_calibrated(oof_stack(CORE_FEATURES), train, calib, ENSEMBLE_FEATURES),
                val,
                ENSEMBLE_FEATURES,
            ),
        ]
    )


# ------------------------------------------------------- D-17: round trip


def test_store_round_trips_an_ensemble(members, matrix, splits, tmp_path):
    """`src/nflpred/modeling/store.py` needed no edits — that is what this proves.

    The 17-column feature tuple is the reason: an ensemble is a
    ``predict_proba``-shaped object over `ENSEMBLE_FEATURES` like any other
    model, so `save_models`/`load_models` take it unchanged.
    """
    _, calib, val = splits
    stack = prefit_stack(members, calib)

    save_models({"ensemble": stack}, "phase5", matrix, ENSEMBLE_FEATURES, root=tmp_path)
    loaded, manifest = load_models("phase5", matrix, root=tmp_path)

    assert manifest["n_features"] == len(ENSEMBLE_FEATURES) == 17
    assert manifest["features"][-1] == "elo_prob"
    assert manifest["splits"] == {"train": 2670, "calib": 801, "val": 821}

    np.testing.assert_allclose(
        home_win_probability(loaded["ensemble"], val, ENSEMBLE_FEATURES),
        home_win_probability(stack, val, ENSEMBLE_FEATURES),
    )


# --------------------------------------------------------- confidence bands


def test_confidence_bands_are_monotone_in_distance_from_a_coin_flip():
    order = [label for _, label in CONFIDENCE_BANDS]
    edges = [0.0, 0.02, 0.05, 0.10, 0.40]
    labels = [confidence(0.5 + e) for e in edges]

    # Every step outward is the same band or a stronger one, never a weaker one.
    positions = [order.index(label) for label in labels]
    assert positions == sorted(positions)
    # Symmetric: an away-side edge reads the same as a home-side one.
    assert [confidence(0.5 - e) for e in edges] == labels


def test_no_read_cut_is_three_points():
    assert confidence(0.50) == "no read"
    assert confidence(0.5 + 0.029) == "no read"
    assert confidence(0.5 + 0.031) == "low"


# ------------------------------------------------------------ record shape


def test_record_matches_the_output_contract(members, splits):
    _, calib, val = splits
    stack = prefit_stack(members, calib)
    records = predict_records(stack, members, val.head(5))

    assert len(records) == 5
    for record in records:
        assert set(record) == {
            "game_id",
            "home_team",
            "away_team",
            "model_votes",
            "ensemble",
            "agreement",
        }
        # Phase 6 adds this; Phase 5 must not fabricate a placeholder for it.
        assert "explanation" not in record

        assert list(record["model_votes"]) == list(MEMBER_ORDER)
        for vote in record["model_votes"].values():
            assert set(vote) == {"pick", "home_win_prob"}
            assert vote["pick"] in {record["home_team"], record["away_team"]}

        assert set(record["ensemble"]) == {"pick", "home_win_prob", "confidence"}
        assert record["ensemble"]["confidence"] in {label for _, label in CONFIDENCE_BANDS}

        agreed, total = record["agreement"].split("/")
        assert int(total) == len(MEMBER_ORDER)
        assert int(agreed) == sum(
            vote["pick"] == record["ensemble"]["pick"]
            for vote in record["model_votes"].values()
        )
