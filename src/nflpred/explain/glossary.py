"""The feature glossary: what each column is called when it is spoken.

Every name a narrative uses comes from here, so no schema name ever reaches a
reader. See the package docstring for how the pieces fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

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
      ``away_backup_qb_starting`` at 1 is good for it - deliberately asymmetric
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
    "net_explosive_r8_diff": _diff("net rate of 20-yard plays over the last 8 games", "pct"),
    "net_points_r8_diff": _diff("net points per game over the last 8 games", "points"),
    "net_third_down_r8_diff": _diff("net third-down rate over the last 8 games", "pct"),
    # Sacks allowed minus sacks generated: positive is the bad direction.
    "net_sack_r8_diff": _diff("net sack rate over the last 8 games", "pct", favours="away"),
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

#: Formatters, one per ``unit``. A magnitude is always spoken unsigned - the
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
