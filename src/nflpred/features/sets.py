"""The named feature sets, and the names of the columns they draw from.

Kept apart from :mod:`nflpred.features.build` so that the modelling, tuning and
explanation layers can say *which* columns they use without importing the code
that builds them. `build` imports the side and matchup constants from here, so
the matrix and the names it is read by cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from nflpred.config import ROLLING_WINDOWS, STD_WINDOW_STATS
from nflpred.features.pbp_agg import OFFENSE_STATS
from nflpred.features.rolling import stem

#: The Phase 2 baseline: seven rolling differentials and three situational
#: columns. Deliberately absent - ``spread_line`` / ``total_line`` (Tier 3,
#: measured separately), turnovers (Tier 3, barely persistent week to week),
#: Elo and the QB composite (Phase 3). This tuple is the bar every later phase
#: has to clear, so it does not change once measured. **Do not edit it.** A bar
#: that moves is not a bar.
BASELINE_FEATURES: Final[tuple[str, ...]] = (
    "net_epa_r4_diff",
    "net_epa_r8_diff",
    "net_success_r8_diff",
    "net_pass_epa_r8_diff",
    "net_rush_epa_r8_diff",
    "net_yards_per_play_r8_diff",
    "net_explosive_r8_diff",
    "rest_diff",
    "div_game",
    "is_dome",
)

#: Elo's contribution, as one column. ``elo_prob`` is deliberately not here:
#: its logit is an exact linear function of ``elo_diff``, so for a logistic
#: regression the two are the same feature. It rides the matrix as the
#: standalone Elo estimator instead.
ELO_FEATURES: Final[tuple[str, ...]] = ("elo_diff",)

#: The Tier 1 QB composite. Form as a differential, the backup flags per side -
#: a backup starting for the home team and one starting for the away team are
#: not the same event, and differencing them would say they were.
QB_FEATURES: Final[tuple[str, ...]] = (
    "qb_epa_r4_diff",
    "qb_epa_r8_diff",
    "qb_cpoe_r8_diff",
    "home_backup_qb_starting",
    "away_backup_qb_starting",
)

#: The 16 columns that minimised validation log loss at Phase 3 - the baseline
#: ten plus Elo plus the QB composite. Named here so Phase 4 can measure every
#: estimator against it as well as against the curated 29.
#:
#: ``BASELINE_FEATURES`` is deliberately untouched by this: it stays the Phase 2
#: bar, and a bar that moves is not a bar.
CORE_FEATURES: Final[tuple[str, ...]] = (*BASELINE_FEATURES, *ELO_FEATURES, *QB_FEATURES)

#: The Phase 3 headline: the curated set, measured against the Phase 2 bar.
PHASE3_FEATURES: Final[tuple[str, ...]] = (
    # --- team form, net differentials
    "net_epa_r4_diff",
    "net_epa_r8_diff",
    "net_epa_std_diff",
    "net_success_r8_diff",
    "net_pass_epa_r8_diff",
    "net_rush_epa_r8_diff",
    "net_yards_per_play_r8_diff",
    "net_explosive_r8_diff",
    "net_points_r8_diff",
    "net_third_down_r8_diff",
    "net_sack_r8_diff",
    # --- the same pass/rush signal, kept as offence and defence separately
    "home_pass_epa_matchup_r8",
    "away_pass_epa_matchup_r8",
    "home_rush_epa_matchup_r8",
    "away_rush_epa_matchup_r8",
    # --- Tier 1
    *ELO_FEATURES,
    *QB_FEATURES,
    # --- situational
    "rest_diff",
    "short_week_diff",
    "off_bye_diff",
    "div_game",
    "is_dome",
    "is_neutral_site",
    "wind",
    # --- Tier 3, shrunk by its measured reliability
    "turnover_diff_shrunk",
)

#: Variants. Each differs from ``PHASE3_FEATURES`` in exactly one way, so the
#: checkpoint table reads as a set of controlled comparisons rather than a
#: collection of unrelated models.
PHASE3_FEATURES_NO_TO: Final[tuple[str, ...]] = tuple(
    f for f in PHASE3_FEATURES if f != "turnover_diff_shrunk"
)
PHASE3_FEATURES_MARKET: Final[tuple[str, ...]] = (
    *PHASE3_FEATURES,
    "spread_line",
    "total_line",
)

#: Statistics kept as separate offence/defence halves rather than only as a net.
MATCHUP_STATS: Final[tuple[str, ...]] = ("pass_epa", "rush_epa")

#: The window the matchup columns use. One window, not both: the point is the
#: offence/defence split, and duplicating it at r4 would double four columns to
#: say the same thing twice.
MATCHUP_WINDOW: Final[int] = 8

#: Per-team columns carried through the home/away split with a side prefix.
SIDE_COLUMNS: Final[tuple[str, ...]] = (
    "elo_pre",
    "qb_epa_r4",
    "qb_epa_r8",
    "qb_cpoe_r4",
    "qb_cpoe_r8",
    "qb_prior_starts",
    "backup_qb_starting",
    "qb_form_unknown",
    *[f"off_{s}_r{MATCHUP_WINDOW}" for s in MATCHUP_STATS],
    *[f"def_{s}_r{MATCHUP_WINDOW}" for s in MATCHUP_STATS],
)

#: Side columns that also get a ``home − away`` differential, mapped to the name
#: that differential takes. ``elo_pre`` differences to plain ``elo_diff``, which
#: is what the rest of the project calls it.
SIDE_DIFFS: Final[dict[str, str]] = {
    "elo_pre": "elo_diff",
    "qb_epa_r4": "qb_epa_r4_diff",
    "qb_epa_r8": "qb_epa_r8_diff",
    "qb_cpoe_r4": "qb_cpoe_r4_diff",
    "qb_cpoe_r8": "qb_cpoe_r8_diff",
}


def net_feature_names(
    windows: Sequence[int] = ROLLING_WINDOWS,
    stats: Sequence[str] = OFFENSE_STATS,
) -> tuple[str, ...]:
    """Names of the net differential columns, in emission order."""
    return tuple(f"net_{stem(s)}_r{w}_diff" for s in stats for w in windows)


def net_std_feature_names(stats: Sequence[str] = STD_WINDOW_STATS) -> tuple[str, ...]:
    """Names of the season-to-date net differential columns."""
    return tuple(f"net_{stem(s)}_std_diff" for s in stats)


def matchup_feature_names() -> tuple[str, ...]:
    """Names of the split offence-against-defence columns."""
    return tuple(
        f"{side}_{s}_matchup_r{MATCHUP_WINDOW}" for side in ("home", "away") for s in MATCHUP_STATS
    )


def wide_features(
    windows: Sequence[int] = ROLLING_WINDOWS,
    stats: Sequence[str] = OFFENSE_STATS,
    std_stats: Sequence[str] = STD_WINDOW_STATS,
) -> tuple[str, ...]:
    """Every legal feature the matrix carries, market columns excluded.

    The kitchen sink, as a control. If ~100 columns beat the curated ~29 by
    enough to matter, the curation was wrong; if they do not, the curation is
    doing its job and the extra columns were noise. Either answer is worth one
    row of the checkpoint table.
    """
    return (
        *net_feature_names(windows, stats),
        *net_std_feature_names(std_stats),
        *matchup_feature_names(),
        *[f"{side}_{c}" for side in ("home", "away") for c in SIDE_COLUMNS],
        *SIDE_DIFFS.values(),
        "rest_diff",
        "home_rest",
        "away_rest",
        "home_short_week",
        "away_short_week",
        "short_week_diff",
        "home_off_bye",
        "away_off_bye",
        "off_bye_diff",
        "div_game",
        "is_dome",
        "is_neutral_site",
        "is_postseason",
        "wind",
        "weather_unknown",
        "std_is_fallback",
        "cpoe_unknown",
        "turnover_diff_shrunk",
    )
