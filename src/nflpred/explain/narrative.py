"""Ranked factors, the confidence decomposition, and the templated narrative.

Everything here describes what the model weights, never what wins games - see
:data:`BANNED_PHRASES`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np

from nflpred.explain.attribution import Attribution
from nflpred.explain.glossary import FEATURE_GLOSSARY, describe
from nflpred.modeling.ensemble import (
    CONFIDENCE_BANDS,
    ENSEMBLE_FEATURES,
    edge,
)

#: A contribution smaller than this in log-odds is not a factor worth speaking.
#: Roughly 0.0025 of probability at the middle of the range - below that, naming
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
    team* - that is the quantity the additivity identity decomposes, so it is
    what the ``shap`` field reports. But "for" and "against" are relative to the
    **pick**, per the spec ("which features drove the pick, which cut against
    it"). On an away pick the two run opposite: a negative contribution is what
    supports the verdict. ``pick_is_home`` is what keeps them straight, and
    without it every away pick - roughly two games in five - would file its
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

    * *"fewer than 4 games of rolling history"* - D-6 runs rolling windows
      across season boundaries, so ``games_in_window_r4`` is 4 for every row in
      the matrix. Structurally dead. The live version of the same concern is
      early-season form being mostly last season's, which is what is reported.
    * *"any feature with a null that had to be imputed"* - the 17 columns carry
      zero nulls in train, calib and validation. The check is kept, so it fires
      if that ever stops being true.

    The third fires often enough to carry the driver on its own: 269 of 821
    validation games have a different-than-usual starter on one side.
    """
    distance = edge(probability)
    pick = record["ensemble"]["pick"]
    # Quoted for the team actually picked, not for the home team. `probability`
    # is P(home win) everywhere in this project, so on an away pick the two are
    # complements - and "favours DEN at 39.0%" is a sentence that reads as a
    # contradiction.
    for_pick = probability if pick == record["home_team"] else 1.0 - probability

    if distance < NO_READ_EDGE:
        signal = f"no read - {probability:.1%} is within {NO_READ_EDGE:.2f} of a coin flip"
    elif distance < CONFIDENCE_BANDS[2][0]:
        # D-21: the two middle bands could not be told apart on validation, so
        # they are presented as one undifferentiated read. The bands themselves
        # are untouched - the collapse is a narrative choice, not a refit.
        signal = (
            f"thin edge - {for_pick:.1%} for {pick}; validation could not "
            f"separate the two middle bands, so they are reported as one"
        )
    else:
        signal = f"clear edge - {for_pick:.1%} for {pick}"

    votes = record["model_votes"]
    agreed = sum(vote["pick"] == pick for vote in votes.values())
    spread = float(np.std([vote["home_win_prob"] for vote in votes.values()]))
    agreement = (
        f"{agreed} of {len(votes)} members pick {pick}; member probabilities spread {spread:.3f}"
    )
    if agreed < len(votes):
        dissenters = [n for n, v in votes.items() if v["pick"] != pick]
        agreement += f" ({', '.join(dissenters)} dissenting)"
    else:
        agreement += " - unanimous, which is the sharper of the two drivers here"

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
            "full - no early-season, quarterback or missing-input caveats"
            if not quality
            else "reduced - " + "; ".join(quality)
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
    0.959, so genuine per-model disagreement is rare - if every game produced a
    rich dissent paragraph, this function would be inventing a diversity the
    ensemble does not have.

    Votes come from the record, so this names exactly the dissenters the record
    shows. The *reason* comes from the member's own attribution, which is of the
    same fitted model before its sigmoid - monotone, so the feature ranking is
    unchanged even though the probability is not.
    """
    pick = record["ensemble"]["pick"]
    home, away = record["home_team"], record["away_team"]
    dissenters = [n for n, v in record["model_votes"].items() if v["pick"] != pick]
    if not dissenters:
        return f"none - all {len(record['model_votes'])} members pick {pick}."

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
    explanation reflects a changed model or just sampling noise - which is the
    whole reason the spec forbids one.

    Every phrase here describes **model behaviour**: what the model weights,
    never what wins football games. :data:`BANNED_PHRASES` is the executable
    form of that constraint.
    """
    pick = record["ensemble"]["pick"]
    home, away = record["home_team"], record["away_team"]
    other = away if pick == home else home

    if edge(probability) < NO_READ_EDGE:
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

    edge_word = "clear" if edge(probability) >= CONFIDENCE_BANDS[2][0] else "thin"
    parts.append(
        f"Confidence is {record['ensemble']['confidence']}: the edge is {edge_word}, and "
        f"{agreed} of {len(votes)} members agree (member probabilities spread {spread:.3f})."
    )

    # The anchors are home-win probabilities, but the sentence above speaks for
    # the pick, so an away pick reads them flipped.
    elo, market = baselines.get("elo_only"), baselines.get("market_implied")

    def for_pick_side(anchor: float) -> float:
        return anchor if pick == home else 1.0 - anchor

    anchors = [
        f"Elo alone {for_pick_side(elo):.1%}" if elo is not None else None,
        f"the market {for_pick_side(market):.1%}" if market is not None else None,
    ]
    parts.append(f"For reference, for {pick}: " + ", ".join(a for a in anchors if a) + ".")

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
