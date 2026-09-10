"""The explanation layer, and the claims that would be wrong *and* convincing.

An attribution bug does not crash. It produces a fluent, confident paragraph
that is backwards — which is strictly worse than no explanation, because a user
cannot tell the difference by reading it. So this file is aimed at the failures
that survive inspection:

**An explainer pointed at the wrong object.** `TreeExplainer` on a
`CalibratedClassifierCV` still returns numbers; they just do not describe the
model that made the pick. Additivity against each member's own raw output is the
only thing that catches it.

**A mislabelled attribution space.** Treating the random forest's probability
output as a log-odds margin (or vice versa) leaves the per-model values intact
and quietly corrupts the combination. The ensemble reconciliation catches it.

**A glossary sign error.** `home_backup_qb_starting` at 1 is bad for the home
team while `away_backup_qb_starting` at 1 is good for it, and five of the 29
columns run opposite to the "positive favours home" default. Get one wrong and
the prose names the wrong team, fluently, forever.

**A causal claim.** SHAP describes what the model weights, never what wins games.

Runs against the real feature matrix and the real fitted models, like
`tests/test_ensemble.py` — a synthetic frame would exercise none of the library
shape differences that :func:`~nflpred.explain._normalise` exists to absorb.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nflpred.explain import (
    _FORMATS,
    ADDITIVITY_ATOL,
    BANNED_PHRASES,
    FEATURE_GLOSSARY,
    NO_READ_EDGE,
    PROBABILITY,
    attribute_all,
    base_logit,
    build_explainers,
    combine,
    describe,
    explained_records,
    meta_terms,
    reference,
    top_factors,
)
from nflpred.features.build import CORE_FEATURES, PHASE3_FEATURES
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

#: The ensemble total reconciles at the float round-trip through sigmoid and
#: back — measured at 1.4e-07 across all 821 validation games — not at machine
#: epsilon. Asserting 1e-9 here would be asserting something untrue about
#: floating point rather than something true about the method.
ENSEMBLE_ATOL = 1e-6


@pytest.fixture(scope="module")
def fitted(models: dict, splits) -> dict:
    """Stack A and its raw members — the headline, and the objects to explain."""
    train, calib, _ = splits
    raw = {name: inner_estimator(model) for name, model in models.items()}
    stack = prefit_stack(build_members(raw), calib)
    coefficients, intercept = meta_terms(stack)
    return {
        "models": models,
        "raw": raw,
        "stack": stack,
        "coefficients": coefficients,
        "intercept": intercept,
        "explainers": build_explainers(raw, float(train["elo_prob"].mean())),
    }


@pytest.fixture(scope="module")
def sample(splits) -> pl.DataFrame:
    """A hundred validation games. Enough for the forest's additivity to mean
    something, small enough that the suite stays quick."""
    _, _, val = splits
    return val.head(100)


@pytest.fixture(scope="module")
def attributed(fitted: dict, sample: pl.DataFrame) -> dict:
    x = sample.select(ENSEMBLE_FEATURES).to_numpy().astype(float)
    attributions = attribute_all(fitted["explainers"], x)
    return {
        "x": x,
        "attributions": attributions,
        "contributions": combine(attributions, fitted["coefficients"]),
        "origin": base_logit(attributions, fitted["coefficients"], fitted["intercept"]),
    }


# ------------------------------------------------------------- additivity

@pytest.mark.parametrize("name", [n for n in MEMBER_ORDER if n != ELO_MEMBER])
def test_each_member_sums_to_its_own_raw_output(name, fitted, attributed):
    """`shap.sum(axis=1) + base_value == the model's own uncalibrated output`.

    The spec's first validation requirement, per model. A member that fails is
    almost always an explainer built on the calibrated wrapper rather than on
    `inner_estimator`'s output — which would still produce plausible numbers.
    """
    attribution = attributed["attributions"][name]
    target = reference(name, fitted["raw"][name], attributed["x"][:, : len(CORE_FEATURES)])

    np.testing.assert_allclose(
        attribution.totals() + attribution.base,
        target,
        atol=ADDITIVITY_ATOL[name],
        err_msg=f"{name} attributions do not sum to its own raw output",
    )


def test_elo_attribution_is_the_rating_itself(attributed, splits):
    """Elo needs no explainer — the rating *is* the explanation."""
    _, _, val = splits
    attribution = attributed["attributions"][ELO_MEMBER]
    assert attribution.space == PROBABILITY
    np.testing.assert_allclose(
        attribution.totals() + attribution.base,
        val.head(100)["elo_prob"].to_numpy(),
        atol=1e-12,
    )


def test_xgboost_needs_a_looser_tolerance_than_the_rest():
    """The tolerance trap, pinned so it cannot be "tidied" into one constant.

    XGBoost returns float32 contributions and reconciles at ~1e-6; every other
    member reconciles at ~1e-15. A single tight tolerance across all five fails
    on xgboost alone, and a single loose one stops testing the other four.
    """
    assert ADDITIVITY_ATOL["xgboost"] > ADDITIVITY_ATOL["lightgbm"] * 1000


def test_the_ensemble_total_reconciles_exactly(fitted, attributed, sample):
    """`sum_j contribution_j == logit(p_ensemble) - base_logit`.

    The secant identity. This is what makes the combination exact rather than a
    Taylor approximation with an error bar — the naive `p(1-p)` slope misses by
    up to 0.43 in log-odds on this data, concentrated in exactly the confident
    games a user cares about.
    """
    probability = home_win_probability(fitted["stack"], sample, ENSEMBLE_FEATURES)
    expected = np.log(probability / (1.0 - probability)) - attributed["origin"]

    np.testing.assert_allclose(
        attributed["contributions"].sum(axis=1), expected, atol=ENSEMBLE_ATOL
    )


def test_a_flat_member_contributes_nothing_rather_than_dividing_by_zero(fitted, attributed):
    """The degenerate case the secant slope has to guard.

    A member whose attributions sum to ~0 has not moved its own probability
    either, so its contribution is zero. Constructed rather than sampled: no
    real validation row happens to hit it, which is exactly why it needs a test.
    """
    from nflpred.explain import MARGIN, Attribution, secant_slopes

    flat = Attribution(np.zeros((3, len(ENSEMBLE_FEATURES))), 0.3, MARGIN)
    slopes = secant_slopes({"flat": flat})
    np.testing.assert_array_equal(slopes["flat"], np.zeros(3))


# --------------------------------------------------------------- glossary

def test_every_feature_has_a_glossary_entry():
    """The spec asks this to fail loudly, so it does.

    Scoped to the union of every *defined* feature set plus `elo_prob`, not to
    the matrix's 126 columns: a glossary entry for a column no model can select
    would be untested prose, and the matrix carries near-twins that would invite
    a copy-pasted, subtly wrong description.
    """
    required = {*CORE_FEATURES, *PHASE3_FEATURES, *ENSEMBLE_FEATURES, "elo_prob"}
    missing = sorted(required - set(FEATURE_GLOSSARY))
    assert not missing, f"FEATURE_GLOSSARY is missing {missing}"


def test_glossary_entries_are_well_formed():
    for name, feature in FEATURE_GLOSSARY.items():
        assert feature.favours in {"home", "away", "neither"}, name
        assert feature.kind in {"diff", "flag", "level"}, name
        assert feature.plain and not feature.plain.endswith("."), name
        assert feature.unit in {"flag", *_FORMATS}, name
        # No schema-shaped token. Not "the column name is absent" — `wind` is
        # both a column and an ordinary English word, and banning it would force
        # a worse phrase to satisfy a test rather than a reader.
        assert "_" not in feature.plain, name


def test_the_five_inverted_features_are_marked_as_such():
    """Measured on the training split, not assumed from the column name.

    Each of these correlates *negatively* with a home win, so a positive value
    favours the away team. They are the glossary's only chance to be
    confidently backwards, and the default is the other direction.
    """
    for name in (
        "net_sack_r8_diff",
        "turnover_diff_shrunk",
        "away_pass_epa_matchup_r8",
        "away_rush_epa_matchup_r8",
        "home_backup_qb_starting",
    ):
        assert FEATURE_GLOSSARY[name].favours == "away", name

    assert FEATURE_GLOSSARY["away_backup_qb_starting"].favours == "home"
    for name in ("div_game", "is_dome", "is_neutral_site", "wind"):
        assert FEATURE_GLOSSARY[name].favours == "neither", name


def test_the_two_elo_columns_are_one_factor():
    """Ranked apart they land in the top three together in three games out of
    four, and the narrative reads "Elo edge, and also Elo edge"."""
    assert (
        FEATURE_GLOSSARY["elo_diff"].key("elo_diff")
        == FEATURE_GLOSSARY["elo_prob"].key("elo_prob")
    )


def test_sign_conventions_put_a_fired_flag_on_the_right_side():
    """The test that catches a backwards glossary entry.

    A backup starting for the *away* team should read as helping the home team
    and a backup starting for the *home* team as hurting it. Driven off
    hand-built contributions rather than a fitted model, so it tests the
    glossary and the ranking, which is what a sign error lives in — not whether
    the boosters happened to learn the right direction this week.
    """
    row = dict.fromkeys(ENSEMBLE_FEATURES, 0.0)
    row["home_backup_qb_starting"] = 1.0
    row["away_backup_qb_starting"] = 1.0

    contributions = np.zeros(len(ENSEMBLE_FEATURES))
    contributions[ENSEMBLE_FEATURES.index("away_backup_qb_starting")] = 0.20
    contributions[ENSEMBLE_FEATURES.index("home_backup_qb_starting")] = -0.20

    factors_for, factors_against = top_factors(contributions, row, "BUF", "KC")

    assert [f["feature"] for f in factors_for] == ["away_backup_qb_starting"]
    assert [f["feature"] for f in factors_against] == ["home_backup_qb_starting"]
    # The phrase must name the team the flag belongs to, not the other one.
    assert factors_for[0]["plain"].startswith("KC")
    assert factors_against[0]["plain"].startswith("BUF")


def test_a_zero_flag_is_not_a_factor():
    """"The away team is not starting a different quarterback" is the absence of
    a factor, not a factor. Trees hand it a non-zero contribution anyway."""
    row = dict.fromkeys(ENSEMBLE_FEATURES, 0.0)
    contributions = np.zeros(len(ENSEMBLE_FEATURES))
    contributions[ENSEMBLE_FEATURES.index("away_backup_qb_starting")] = 0.5

    factors_for, factors_against = top_factors(contributions, row, "BUF", "KC")
    assert factors_for == []
    assert factors_against == []


def test_describe_never_emits_a_column_name(splits):
    _, _, val = splits
    row = val.head(1).to_dicts()[0]
    for name in ENSEMBLE_FEATURES:
        phrase = describe(name, float(row[name]), "BUF", "KC")
        assert name not in phrase
        assert "_" not in phrase, f"{name} produced {phrase!r}"


# ------------------------------------------------- the generated paragraphs

@pytest.fixture(scope="module")
def explained(fitted, splits, sample) -> list[dict]:
    _, calib, _ = splits
    models, stack = fitted["models"], fitted["stack"]
    platt = fit_elo_platt(calib)
    probability = home_win_probability(stack, sample, ENSEMBLE_FEATURES)
    records = predict_records(stack, build_members(models, platt=platt), sample)
    return explained_records(
        records,
        sample,
        fitted["explainers"],
        fitted["coefficients"],
        fitted["intercept"],
        probability,
    )


def _strings(node) -> list[str]:
    """Every user-facing string in a record, at any depth."""
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        return [s for key, value in node.items() if key != "feature" for s in _strings(value)]
    if isinstance(node, list):
        return [s for item in node for s in _strings(item)]
    return []


def test_no_narrative_makes_a_causal_claim(explained):
    """SHAP describes model behaviour. "The model weights X heavily", never
    "X causes wins" — the spec is explicit and the template must not drift."""
    for record in explained:
        for text in _strings(record["explanation"]):
            lowered = text.lower()
            for phrase in BANNED_PHRASES:
                assert phrase not in lowered, f"{phrase!r} in {text!r}"


def test_no_raw_column_name_reaches_a_user_facing_string(explained):
    """`feature` keys are machine fields and may carry column names; `plain`,
    `narrative` and the driver strings may not."""
    for record in explained:
        for text in _strings(record["explanation"]):
            for name in ENSEMBLE_FEATURES:
                assert name not in text, f"{name!r} leaked into {text!r}"


def test_a_no_read_game_refuses_to_explain_itself(explained):
    """The honesty constraint, executable.

    Inside 0.03 of a coin flip the model says it has no meaningful read and
    assembles no ranked rationale, rather than manufacturing one.
    """
    no_read = [
        r
        for r in explained
        if abs(r["ensemble"]["home_win_prob"] - 0.5) < NO_READ_EDGE
    ]
    assert no_read, "no near-coin-flip games in the sample — the constraint is untested"

    for record in no_read:
        explanation = record["explanation"]
        assert explanation["top_factors_for"] == []
        assert explanation["top_factors_against"] == []
        assert "no meaningful read" in explanation["narrative"]


def test_a_confident_game_does_assemble_a_rationale(explained):
    """The other half: the no-read rule must not have silenced everything."""
    confident = [
        r for r in explained if abs(r["ensemble"]["home_win_prob"] - 0.5) > 0.13
    ]
    assert confident
    assert all(r["explanation"]["top_factors_for"] for r in confident)


def test_the_stated_top_factor_is_not_globally_ignored(explained, attributed):
    """The spec's second validation requirement.

    Scoped to "some member that can actually see the column weights it", which
    is the only formulation that survives contact with the data: the boosters
    are minimum-capacity by D-16 and xgboost never splits on `is_dome` or either
    backup flag, while `elo_prob` is invisible to five of the six members by
    column layout rather than by indifference.
    """
    visible = {
        name: np.abs(attributed["attributions"][name].values).mean(axis=0)
        for name in MEMBER_ORDER
    }
    for record in explained:
        factors = record["explanation"]["top_factors_for"]
        factors += record["explanation"]["top_factors_against"]
        for factor in factors:
            position = ENSEMBLE_FEATURES.index(factor["feature"])
            assert any(
                importance[position] > 0 for importance in visible.values()
            ), f"{factor['feature']} is globally ignored by every member"


def test_the_record_carries_the_spec_s_explanation_keys(explained):
    for record in explained:
        explanation = record["explanation"]
        assert set(explanation) == {
            "top_factors_for",
            "top_factors_against",
            "confidence_drivers",
            "dissent",
            "vs_baselines",
            "narrative",
            "per_model",
            "method",
        }
        assert set(explanation["confidence_drivers"]) == {
            "signal_strength",
            "model_agreement",
            "data_quality",
        }
        assert set(explanation["vs_baselines"]) == {"elo_only", "market_implied"}
        # The spec's step 2 requires the combination be labelled an
        # approximation, and step 3 requires the per-model values underneath.
        assert "approximation" in explanation["method"]["combination"]
        assert set(explanation["per_model"]) == set(MEMBER_ORDER)
        for factor in explanation["top_factors_for"] + explanation["top_factors_against"]:
            assert set(factor) == {"feature", "value", "shap", "plain"}


def test_per_game_reconciliation_is_reported_and_tiny(explained):
    """The approximation label is backed by a number, not a disclaimer."""
    for record in explained:
        assert record["explanation"]["method"]["reconciliation"] < ENSEMBLE_ATOL


# ------------------------------------------------------------ determinism

def test_the_same_game_explains_identically_twice(fitted, sample):
    """Template-based and deterministic, per the spec — no LLM, and no RNG.

    The live risk is `feature_perturbation="interventional"`, which sub-samples
    a background set and would make the same game explain differently from week
    to week. `build_explainers` pins `tree_path_dependent` for this reason.
    """
    import json

    x = sample.head(5).select(ENSEMBLE_FEATURES).to_numpy().astype(float)
    runs = []
    for _ in range(2):
        attributions = attribute_all(fitted["explainers"], x)
        contributions = combine(attributions, fitted["coefficients"])
        runs.append(json.dumps(contributions.tolist()))
    assert runs[0] == runs[1]


def test_a_tie_gets_one_explanation_not_two(splits, explained):
    """D-4 expands a tie into two rows for *fitting*; it still gets one
    prediction, and therefore one explanation."""
    _, _, val = splits
    assert val.height == val["game_id"].n_unique() == 821
    assert sorted(val.filter(pl.col("is_tie") == 1)["game_id"].to_list()) == [
        "2019_01_DET_ARI",
        "2020_03_CIN_PHI",
        "2021_10_DET_PIT",
    ]

    ids = [r["game_id"] for r in explained]
    assert len(ids) == len(set(ids))


# ------------------------------------------------------------ split hygiene
#
# The static "script never names the unseen split" check for every phase lives
# in test_phase_hygiene.py now. This is its runtime half for Phase 6.


def test_explanations_ignore_the_held_out_split(matrix, splits):
    """Runtime half: poison every held-out row and no attribution moves."""
    _, _, val = splits
    poisoned = matrix.with_columns(
        [
            pl.when(pl.col("split") == "test").then(999.0).otherwise(pl.col(c)).alias(c)
            for c in ENSEMBLE_FEATURES
        ]
    )
    assert poisoned.filter(pl.col("split") == "test").height

    def contributions(source: pl.DataFrame) -> np.ndarray:
        models = fit_all(source, CORE_FEATURES)
        raw = {name: inner_estimator(model) for name, model in models.items()}
        stack = prefit_stack(build_members(raw), split_frame(source, "calib"))
        coefficients, _ = meta_terms(stack)
        explainers = build_explainers(raw, float(split_frame(source, "train")["elo_prob"].mean()))
        x = val.head(50).select(ENSEMBLE_FEATURES).to_numpy().astype(float)
        return combine(attribute_all(explainers, x), coefficients)

    np.testing.assert_array_equal(
        contributions(matrix),
        contributions(poisoned),
        err_msg="an attribution changed when the held-out split was rewritten",
    )
