"""The Phase 4 model layer, with the two claims that are easy to get wrong.

Most of what `src/nflpred/modeling/base.py` does would be *visibly* broken if it were
broken — a model that failed to fit raises. Two things would not be visible:

**Sample weights that go nowhere.** D-4 encodes a tie as two half-weight rows,
and a `Pipeline` silently ignores a ``sample_weight`` passed to the wrong place.
An estimator that dropped the weights would fit fine, score fine, and quietly
treat every tie as two whole games. So each of the five is made to prove that
changing the weights changes its predictions.

**Artifacts that no longer match their definitions.** D-17 makes model files
outputs rather than a cache, which is only meaningful if a stale one is caught.
The round-trip test corrupts a fingerprint on purpose and demands a failure.

Runs against the real feature matrix. Synthetic frames would not exercise the
splits, the ties, or the null columns that make this layer interesting.
"""

from __future__ import annotations

import ast
import json
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nflpred.evaluate import ACCURACY_CEILING, expand_ties, halt_if_suspicious
from nflpred.features.build import CORE_FEATURES, to_training_arrays
from nflpred.modeling.base import (
    BUILDERS,
    build_models,
    fit_all,
    fit_calibrated,
    fit_weighted,
    home_win_probability,
    inner_estimator,
)
from nflpred.modeling.store import MANIFEST_NAME, fingerprint, load_models, save_models

PHASE4_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "phases" / "phase4_models.py"


pytestmark = pytest.mark.data

@pytest.fixture(scope="module")
def models(matrix: pl.DataFrame) -> dict:
    return fit_all(matrix, CORE_FEATURES)


# --------------------------------------------------------------- D-4: ties


def test_tie_expands_to_two_half_weight_rows(splits):
    """A tie game enters training as two rows of half weight, one per outcome."""
    train, _, _ = splits
    ties = train.filter(pl.col("home_win") == 0.5)
    assert ties.height, "the training split should contain at least one tie"

    x, y, w = to_training_arrays(ties, CORE_FEATURES)

    assert x.shape == (2 * ties.height, len(CORE_FEATURES))
    assert sorted(y.tolist()) == [0.0] * ties.height + [1.0] * ties.height
    assert np.allclose(w, 0.5)
    # A tie contributes exactly one game's worth of weight, not two.
    assert w.sum() == pytest.approx(float(ties.height))

    # Both rows of a tie carry that game's features, unchanged.
    first = ties.select(CORE_FEATURES).to_numpy().astype(float)[0]
    assert np.array_equal(x[0], first)
    assert np.array_equal(x[ties.height], first)


def test_expand_ties_composes_with_an_outer_weight():
    """Weights multiply, which is the channel Phase 7's recency weighting uses."""
    _, y, w = expand_ties(np.array([1.0, 0.5, 0.0]), np.array([2.0, 4.0, 1.0]))
    assert sorted(w.tolist()) == [1.0, 2.0, 2.0, 2.0]
    assert sorted(y.tolist()) == [0.0, 0.0, 1.0, 1.0]


@pytest.mark.parametrize("name", list(BUILDERS))
def test_every_estimator_honours_sample_weight(name, splits):
    """D-4's discharge: weights reaching each of the five changes what it learns.

    A `Pipeline` routes ``sample_weight`` by step name and silently ignores it
    otherwise, so "the estimator accepted the argument" is not the claim worth
    testing. The claim is that the argument had an effect.
    """
    train, _, _ = splits
    sample = train.head(600)
    x, y, w = to_training_arrays(sample, CORE_FEATURES)

    lopsided = w.copy()
    lopsided[len(lopsided) // 2 :] *= 0.001

    evenly = fit_weighted(build_models(CORE_FEATURES)[name], x, y, w)
    evenly_probabilities = evenly.predict_proba(x)[:, 1]

    unevenly = fit_weighted(build_models(CORE_FEATURES)[name], x, y, lopsided)
    unevenly_probabilities = unevenly.predict_proba(x)[:, 1]

    assert not np.allclose(evenly_probabilities, unevenly_probabilities), (
        f"{name} produced identical predictions with and without a 1000x weight "
        f"imbalance — sample_weight is being dropped, and every tie in the "
        f"training set is being counted as two whole games."
    )


# ------------------------------------------------------- D-5 / D-13: shape


@pytest.mark.parametrize("name", list(BUILDERS))
def test_one_inner_estimator_per_model(name, models):
    """The property D-5 wanted from `cv='prefit'`, preserved by FrozenEstimator.

    Phase 6 explains one model per prediction. If the calibrator ever fits by
    internal cross-validation instead, this becomes *k* estimators and the SHAP
    layer has nothing single to point at.
    """
    assert len(models[name].calibrated_classifiers_) == 1


@pytest.mark.parametrize("name", list(BUILDERS))
def test_inner_estimator_returns_the_fitted_base(name, models, splits):
    """`inner_estimator` reaches through Frozen wrapper to the fitted estimator."""
    _, _, val = splits
    base = inner_estimator(models[name])

    assert type(base) is type(build_models(CORE_FEATURES)[name])
    # Fitted, and usable on its own — Phase 6 scores it without the calibrator.
    raw = base.predict_proba(val.select(CORE_FEATURES).to_numpy().astype(float))[:, 1]
    assert raw.shape == (val.height,)
    assert ((raw > 0) & (raw < 1)).all()


def test_calibrating_a_frozen_base_is_quiet(splits):
    """The benign FrozenEstimator/sample_weight warning stays suppressed.

    Passing weights to a frozen base warns that they reached only the
    calibrator, which is exactly the intent here. Suppressed narrowly at the one
    call site, and pinned so that a *different* warning would still be heard.
    """
    train, calib, _ = splits
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fit_calibrated(
            build_models(CORE_FEATURES)["logreg"],
            train.head(400),
            calib.head(200),
            CORE_FEATURES,
        )


def test_calibrator_is_fitted_on_the_calibration_split_only(splits):
    """Feeding a different calibration split moves the calibrated probabilities
    but leaves the base untouched — the two fits are genuinely separate."""
    train, calib, val = splits
    estimator = build_models(CORE_FEATURES)["logreg"]

    first = fit_calibrated(estimator, train, calib.head(300), CORE_FEATURES)
    second = fit_calibrated(estimator, train, calib.tail(300), CORE_FEATURES)

    base_first = inner_estimator(first).predict_proba(
        val.select(CORE_FEATURES).to_numpy().astype(float)
    )[:, 1]
    base_second = inner_estimator(second).predict_proba(
        val.select(CORE_FEATURES).to_numpy().astype(float)
    )[:, 1]

    assert np.allclose(base_first, base_second)
    assert not np.allclose(
        home_win_probability(first, val, CORE_FEATURES),
        home_win_probability(second, val, CORE_FEATURES),
    )


# ------------------------------------------------------------ determinism


def test_fit_all_is_reproducible(matrix, splits):
    """Two runs, identical validation probabilities.

    Not a nicety: D-17's fingerprint is only a check if refitting is
    deterministic. Every estimator gets a fixed seed and a fixed thread count,
    since thread count changes the order floats are summed in.
    """
    _, _, val = splits
    first = fit_all(matrix, CORE_FEATURES)
    second = fit_all(matrix, CORE_FEATURES)

    for name in first:
        np.testing.assert_array_equal(
            home_win_probability(first[name], val, CORE_FEATURES),
            home_win_probability(second[name], val, CORE_FEATURES),
            err_msg=f"{name} is not reproducible across two identical fits",
        )


# ------------------------------------------------------ D-17: round trip


def test_save_then_load_round_trips(models, matrix, tmp_path):
    loaded, manifest = _round_trip(models, matrix, tmp_path)

    assert sorted(loaded) == sorted(models)
    assert manifest["features"] == list(CORE_FEATURES)
    assert manifest["n_features"] == len(CORE_FEATURES)
    assert manifest["splits"] == {"train": 2670, "calib": 801, "val": 821}

    for name in models:
        assert fingerprint(loaded[name], matrix, CORE_FEATURES) == manifest["fingerprint"][name]


def test_corrupted_fingerprint_raises(models, matrix, tmp_path):
    """A model that no longer predicts what it did when saved must not load."""
    save_models(models, "phase4", matrix, CORE_FEATURES, root=tmp_path)

    manifest_path = tmp_path / "phase4" / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fingerprint"]["logreg"][0] += 0.05
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_models("phase4", matrix, root=tmp_path)


def test_missing_manifest_raises(matrix, tmp_path):
    with pytest.raises(FileNotFoundError):
        load_models("never-written", matrix, root=tmp_path)


def _round_trip(models, matrix, tmp_path):
    save_models(models, "phase4", matrix, CORE_FEATURES, root=tmp_path)
    return load_models("phase4", matrix, root=tmp_path)


# ------------------------------------------------------------ split hygiene


def test_phase4_script_never_names_the_test_split():
    """Static half of the claim: the checkpoint script cannot ask for test rows.

    Every split is read by name. If ``"test"`` is not a string constant in the
    module, no code path in it can select a test row.
    """
    tree = ast.parse(PHASE4_SCRIPT.read_text(encoding="utf-8"))

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


def test_fitting_ignores_the_test_split(matrix, splits):
    """Runtime half: poison every test row and nothing downstream moves.

    Stronger than reading the code — it would catch a join, a global mean, or a
    scaler that pulled the whole matrix in without asking for a split.
    """
    _, _, val = splits
    poisoned = matrix.with_columns(
        [
            pl.when(pl.col("split") == "test").then(999.0).otherwise(pl.col(c)).alias(c)
            for c in CORE_FEATURES
        ]
        + [
            pl.when(pl.col("split") == "test")
            .then(1.0)
            .otherwise(pl.col("home_win"))
            .alias("home_win")
        ]
    )
    assert poisoned.filter(pl.col("split") == "test").height, "matrix must contain test rows"

    clean_models = fit_all(matrix, CORE_FEATURES)
    poisoned_models = fit_all(poisoned, CORE_FEATURES)

    for name in clean_models:
        np.testing.assert_array_equal(
            home_win_probability(clean_models[name], val, CORE_FEATURES),
            home_win_probability(poisoned_models[name], val, CORE_FEATURES),
            err_msg=f"{name} changed when the test split was rewritten — it is reading test rows",
        )


# ---------------------------------------------------------------- tripwire


def test_tripwire_halts_on_suspicious_accuracy():
    """Constraint 3, executable. 75% straight up is a leak, not a model."""
    table = pl.DataFrame(
        {
            "model": ["plausible", "suspicious"],
            "accuracy": [0.6413, 0.75],
            "log_loss": [0.6398, 0.4],
            "brier": [0.2242, 0.15],
        }
    )
    with pytest.raises(SystemExit, match="suspicious"):
        halt_if_suspicious(table)


def test_tripwire_passes_a_plausible_table():
    table = pl.DataFrame({"model": ["logreg"], "accuracy": [ACCURACY_CEILING - 0.01]})
    halt_if_suspicious(table)


def test_tripwire_exempts_the_market():
    """The market is allowed to be good; that is what makes it the reference."""
    table = pl.DataFrame({"model": ["market (de-vigged)"], "accuracy": [0.98]})
    halt_if_suspicious(table)


def _entry(name: str, accuracy: float, n: int = 100_000):
    """A synthetic (name, y, p) whose straight-up accuracy is exactly ``accuracy``."""
    hits = round(accuracy * n)
    y = np.ones(n)
    p = np.array([0.9] * hits + [0.1] * (n - hits))  # >= 0.5 predicts a home win
    return name, y, p


def test_tripwire_catches_an_accuracy_that_rounds_down_to_the_ceiling():
    """0.72004 displays as 0.7200; passing ``entries`` checks the unrounded value."""
    from nflpred.evaluate import metrics_table

    entries = [_entry("leaky", 0.72004)]
    table = metrics_table(entries)
    assert table["accuracy"][0] == ACCURACY_CEILING  # rounded display hides the leak

    halt_if_suspicious(table)  # rounded column: does not trip (the old behaviour)
    with pytest.raises(SystemExit, match="leaky"):
        halt_if_suspicious(table, entries=entries)  # unrounded: trips


def test_tripwire_with_entries_still_passes_a_genuinely_sub_ceiling_model():
    from nflpred.evaluate import metrics_table

    entries = [_entry("fine", 0.71996)]  # also rounds to 0.7200 for display
    halt_if_suspicious(metrics_table(entries), entries=entries)
