"""Assembles the game-level feature matrix.

Two team rows become one game row, always from the **home team's perspective**.
Home-field advantage is therefore the model's intercept rather than a feature,
which is both simpler and harder to get wrong than carrying a ``is_home``
column on a frame where every row is a home row. Every differential is built
``home − away``, without exception, so the sign of a coefficient always means
the same thing.

Most team statistics enter as a single net-quality differential::

    net_X_diff = (home_off_X - home_def_X) - (away_off_X - away_def_X)

Uniform, and it halves the column count the way the spec's feature-engineering
note asks for.

Pass and rush EPA additionally enter **split into their two halves**::

    home_pass_epa_matchup_r8 = home_off_pass_epa_r8 - away_def_pass_epa_r8
    away_pass_epa_matchup_r8 = away_off_pass_epa_r8 - home_def_pass_epa_r8

This is the "matchup framing" the Phase 2 docstring promised, and it is worth
stating what it does *not* buy. For a single symmetric statistic the matchup
differential is algebraically identical to the net one::

    (home_off - home_def) - (away_off - away_def)
      ==  (home_off - away_def) - (away_off - home_def)

So re-expressing ``net_X_diff`` as a matchup differential adds exactly nothing.
What adds something is keeping the two halves as **separate columns**, which
lets the model weight a good offence differently from a good defence instead of
being forced to treat them as interchangeable. That is what is built here.

Situational columns are taken from ``schedules`` rather than recomputed -
``home_rest``, ``div_game``, ``roof`` and friends are given, and re-deriving
them from play-by-play would only introduce disagreement.

``spread_line`` and ``total_line`` are carried in the matrix but are **not** in
``PHASE3_FEATURES``. They are Tier 3: the strongest single predictors available
and a wholesale import of the market's opinion. ``PHASE3_FEATURES_MARKET``
measures their contribution as a separate variant, which is what the spec asks
for.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import polars as pl

from nflpred.config import (
    FEATURE_MATRIX_PATH,
    MATRIX_START_SEASON,
    ROLLING_WINDOWS,
    STD_WINDOW_STATS,
    TEAM_GAME_PATH,
    TRAIN_SEASONS,
    TURNOVER_RELIABILITY,
    season_split,
)
from nflpred.evaluate import expand_ties
from nflpred.features.elo import build_elo
from nflpred.features.pbp_agg import OFFENSE_STATS
from nflpred.features.qb import build_qb
from nflpred.features.rolling import build_rolling, stem

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

#: Situational columns lifted straight from ``schedules``.
_SITUATIONAL: Final[tuple[str, ...]] = (
    "home_rest",
    "away_rest",
    "div_game",
    "roof",
    "temp",
    "wind",
    "location",
    "spread_line",
    "total_line",
    "home_moneyline",
    "away_moneyline",
)

#: ``roof`` values that mean the game was played indoors. ``open`` is a
#: retractable roof left open, i.e. outdoors on the day.
_INDOOR_ROOFS: Final[frozenset[str]] = frozenset({"dome", "closed"})

#: What a null ``roof`` is assumed to be, and the measurement behind it (D-32).
#:
#: A fixed dome reads ``dome`` in the schedule years ahead. A **retractable**
#: one reads ``open`` or ``closed``, which is a decision made on the day, so it
#: is null for every future game - 43 of the 2026 season's 272, all at the five
#: retractable venues. That makes ``is_dome`` the one situational feature that
#: is not knowable before kickoff, and it is in `BASELINE_FEATURES`.
#:
#: Closed is not a guess: it is the modal state at every one of the five, by a
#: wide margin, across 2006-2025.
#:
#:     stadium          closed  open   closed share
#:     ATL97 (ATL)         56     18      0.76
#:     DAL00 (DAL)        136      9      0.94
#:     HOU00 (HOU)        144     28      0.84
#:     IND00 (IND)        119     34      0.78
#:     PHO00 (ARI)        151     21      0.88
#:     all retractables   606    110      0.85
#:
#: ``roof_unknown`` records where the default was applied. It is deliberately
#: **not** a feature: it is identically zero across every completed game in the
#: matrix, so as a column for a model to learn from it carries nothing at all.
#: It exists so that a live prediction can say which of its inputs was observed.
_ROOF_DEFAULT: Final[str] = "closed"

#: Rest days at or below this are a short week - a Thursday game after a Sunday.
SHORT_WEEK_REST: Final[int] = 4

#: Rest days at or above this mean the team is coming off a bye.
OFF_BYE_REST: Final[int] = 13

#: Statistics kept as separate offence/defence halves rather than only as a net.
_MATCHUP_STATS: Final[tuple[str, ...]] = ("pass_epa", "rush_epa")

#: The window the matchup columns use. One window, not both: the point is the
#: offence/defence split, and duplicating it at r4 would double four columns to
#: say the same thing twice.
_MATCHUP_WINDOW: Final[int] = 8

#: Per-team columns carried through the home/away split with a side prefix.
_SIDE_COLUMNS: Final[tuple[str, ...]] = (
    "elo_pre",
    "qb_epa_r4",
    "qb_epa_r8",
    "qb_cpoe_r4",
    "qb_cpoe_r8",
    "qb_prior_starts",
    "backup_qb_starting",
    "qb_form_unknown",
    *[f"off_{s}_r{_MATCHUP_WINDOW}" for s in _MATCHUP_STATS],
    *[f"def_{s}_r{_MATCHUP_WINDOW}" for s in _MATCHUP_STATS],
)

#: Per-team identifiers carried through the split. Not features - they are what
#: makes a Phase 6 explanation able to name the quarterback it is talking about.
_SIDE_LABELS: Final[tuple[str, ...]] = ("projected_qb",)

#: Side columns that also get a ``home − away`` differential, mapped to the name
#: that differential takes. ``elo_pre`` differences to plain ``elo_diff``, which
#: is what the rest of the project calls it.
_SIDE_DIFFS: Final[dict[str, str]] = {
    "elo_pre": "elo_diff",
    "qb_epa_r4": "qb_epa_r4_diff",
    "qb_epa_r8": "qb_epa_r8_diff",
    "qb_cpoe_r4": "qb_cpoe_r4_diff",
    "qb_cpoe_r8": "qb_cpoe_r8_diff",
}


#: Week numbers never exceed this, so ``season * _WEEK_BASE + week`` is a
#: monotone encoding of the calendar and its dense rank is a clean index.
_WEEK_BASE: Final[int] = 100


def league_week_index() -> pl.Expr:
    """A chronological integer over the matrix's distinct ``(season, week)`` pairs.

    Counted in **league weeks, not calendar weeks**: consecutive indices are
    consecutive weeks *of football*, and the ~30 weeks between a Super Bowl and
    the following September count as one step, not as thirty. This is the one
    place a reader could reasonably assume the other reading, so it is stated
    here rather than left to inference.

    It matters because it is the clock the Phase 7 recency weighting runs on
    (:func:`nflpred.backtest.recency_weights`). Under this definition the spec's
    suggested half-life range of 20-40 weeks is roughly one to two **seasons**;
    under a calendar reading the same numbers would be well under a single
    season, and the grid would be searching a different question. Postseason
    weeks are included and are ordinary steps - a conference championship is one
    week after a divisional round in both readings.

    Dense-ranked rather than counted from a fixed origin, so the index starts at
    1 at :data:`~nflpred.config.MATRIX_START_SEASON` week 1 and has no gaps for the
    weeks a season does not have.
    """
    return (pl.col("season") * _WEEK_BASE + pl.col("week")).rank("dense").cast(pl.Int32)


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
        f"{side}_{s}_matchup_r{_MATCHUP_WINDOW}"
        for side in ("home", "away")
        for s in _MATCHUP_STATS
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
        *[f"{side}_{c}" for side in ("home", "away") for c in _SIDE_COLUMNS],
        *_SIDE_DIFFS.values(),
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


def turnover_reliability(
    team_game: pl.DataFrame, seasons: tuple[int, int] = TRAIN_SEASONS
) -> float:
    """Split-half reliability of a team's per-game turnover margin (Tier 3).

    The spec's instruction for turnovers is "do not let the model lean on raw
    recent turnover margin". A measured shrinkage factor is the honest form of
    that instruction: rather than asserting turnovers are noisy, measure *how*
    noisy and scale the feature by what survives.

    Each team-season's games are split into odd and even halves, each half's
    mean turnover margin is taken, and the two are correlated across all
    team-seasons in ``seasons``. A regular season splits into two halves of
    about eight games each, which is exactly the length of the ``r8`` window the
    factor is applied to - so this correlation estimates the reliability of the
    feature as built, with no Spearman-Brown extrapolation needed.

    Computed on **training seasons only**, like any other fitted quantity - and
    then frozen into :data:`~nflpred.config.TURNOVER_RELIABILITY` rather than
    applied directly, so that the feature does not silently rescale itself every
    time the table behind it changes. This function is what produced that
    constant and what re-verifies it; ``main`` prints both side by side.

    One caveat, stated rather than glossed: multiplying a feature by a constant
    is a **no-op** under a standardised linear model, which rescales it straight
    back. So ``turnover_diff_shrunk`` cannot change the Phase 3 numbers, and it
    is not what answers the question - the ``PHASE3_FEATURES_NO_TO`` variant is,
    and it says turnovers are worth ~0.0002 of validation log loss. The
    shrinkage earns its place from Phase 4 on, where tree splits and
    regularisation paths do respond to scale, and as a statement of how much of
    this column anyone should believe.
    """
    low, high = seasons
    frame = (
        team_game.filter(pl.col("season").is_between(low, high))
        .sort("team", "season", "gameday", "game_id")
        .with_columns(
            # Takeaways minus giveaways: the opponent's offence turned it over
            # (`def_turnovers`) less our own offence doing so (`off_turnovers`).
            turnover_margin=pl.col("def_turnovers") - pl.col("off_turnovers"),
            half=pl.int_range(pl.len()).over(["team", "season"]) % 2,
        )
        .group_by("team", "season", "half")
        .agg(pl.col("turnover_margin").mean())
        .pivot(on="half", index=["team", "season"], values="turnover_margin")
        .drop_nulls()
    )

    return float(frame.select(pl.corr("0", "1")).item())


def _net_quality(
    rolling: pl.DataFrame,
    windows: Sequence[int],
    stats: Sequence[str],
    std_stats: Sequence[str],
) -> pl.DataFrame:
    """Collapse each team-game row's offence and defence into one net figure.

    Defensive columns are what the *opponent* did, so lower is better and the
    subtraction is the right way round: a team with +0.10 offensive EPA/play and
    -0.05 allowed nets +0.15.

    Turnovers are the one statistic where that reading inverts - ``off_turnovers``
    is giveaways, so ``net_turnovers`` is a turnover *deficit*, not a surplus.
    The sign lives in the coefficient; the construction stays uniform.
    """
    return rolling.with_columns(
        [
            (pl.col(f"off_{stem(s)}_r{w}") - pl.col(f"def_{stem(s)}_r{w}")).alias(
                f"net_{stem(s)}_r{w}"
            )
            for s in stats
            for w in windows
        ]
        + [
            (pl.col(f"off_{stem(s)}_std") - pl.col(f"def_{stem(s)}_std")).alias(
                f"net_{stem(s)}_std"
            )
            for s in std_stats
        ]
    )


def build_features(
    team_game: pl.DataFrame,
    schedules: pl.DataFrame,
    pbp: pl.DataFrame,
    windows: Sequence[int] = ROLLING_WINDOWS,
    stats: Sequence[str] = OFFENSE_STATS,
    std_stats: Sequence[str] = STD_WINDOW_STATS,
) -> pl.DataFrame:
    """One row per game, home perspective, features knowable before kickoff."""
    rolling = build_rolling(team_game, windows, stats, std_stats)
    nets = _net_quality(rolling, windows, stats, std_stats)

    elo = build_elo(schedules).select("game_id", "team", "elo_pre", "elo_prob")
    qb = build_qb(pbp, team_game, windows).drop("season", "week", "gameday")

    teams = nets.join(elo, on=["game_id", "team"], how="left").join(
        qb, on=["game_id", "team"], how="left"
    )

    net_cols = [f"net_{stem(s)}_r{w}" for s in stats for w in windows]
    net_cols += [f"net_{stem(s)}_std" for s in std_stats]
    window_cols = [f"games_in_window_r{w}" for w in windows]

    home = teams.filter(pl.col("is_home") == 1).select(
        "game_id",
        "season",
        "week",
        "game_type",
        "gameday",
        "is_neutral_site",
        "is_postseason",
        "std_is_fallback",
        home_team=pl.col("team"),
        away_team=pl.col("opponent"),
        home_win=pl.col("won"),
        elo_prob=pl.col("elo_prob"),
        **{f"home_{c}": pl.col(c) for c in [*window_cols, *_SIDE_COLUMNS, *_SIDE_LABELS]},
        **{c: pl.col(c) for c in net_cols},
    )
    away = teams.filter(pl.col("is_home") == 0).select(
        "game_id",
        **{f"away_{c}": pl.col(c) for c in [*window_cols, *_SIDE_COLUMNS, *_SIDE_LABELS]},
        **{f"{c}_away": pl.col(c) for c in net_cols},
    )

    situational = schedules.select("game_id", *_SITUATIONAL)

    games = (
        home.join(away, on="game_id", how="inner")
        .join(situational, on="game_id", how="left")
        .with_columns(
            # Every differential is home minus away, without exception.
            [(pl.col(c) - pl.col(f"{c}_away")).alias(f"{c}_diff") for c in net_cols]
            + [
                (pl.col(f"home_{c}") - pl.col(f"away_{c}")).alias(name)
                for c, name in _SIDE_DIFFS.items()
            ]
            # The offence/defence split: each side's offence against the other
            # side's defence, kept as its own column.
            + [
                (
                    pl.col(f"{side}_off_{s}_r{_MATCHUP_WINDOW}")
                    - pl.col(f"{other}_def_{s}_r{_MATCHUP_WINDOW}")
                ).alias(f"{side}_{s}_matchup_r{_MATCHUP_WINDOW}")
                for side, other in (("home", "away"), ("away", "home"))
                for s in _MATCHUP_STATS
            ]
        )
        .with_columns(
            rest_diff=(pl.col("home_rest") - pl.col("away_rest")).cast(pl.Int32),
            home_short_week=(pl.col("home_rest") <= SHORT_WEEK_REST).cast(pl.Int8),
            away_short_week=(pl.col("away_rest") <= SHORT_WEEK_REST).cast(pl.Int8),
            home_off_bye=(pl.col("home_rest") >= OFF_BYE_REST).cast(pl.Int8),
            away_off_bye=(pl.col("away_rest") >= OFF_BYE_REST).cast(pl.Int8),
            div_game=pl.col("div_game").cast(pl.Int8),
            # A retractable roof has no recorded position until the day, so a
            # future fixture's `roof` is null. Defaulted to its measured modal
            # state rather than left null, which would drop the game from every
            # fit, and flagged rather than silently filled (see _ROOF_DEFAULT).
            roof_unknown=pl.col("roof").is_null().cast(pl.Int8),
            is_dome=pl.col("roof")
            .fill_null(_ROOF_DEFAULT)
            .is_in(_INDOOR_ROOFS)
            .cast(pl.Int8),
            # `location` is authoritative on neutral sites; the team-game table
            # derives its flag from the same column, so this only re-states it.
            # Promoted from a recorded fact to a feature in Phase 3: a neutral
            # site has a reduced home-field effect, and giving the model the
            # flag lets it learn how reduced instead of forcing a choice between
            # dropping the games and pretending they are ordinary.
            is_neutral_site=pl.col("is_neutral_site").cast(pl.Int8),
            # A tie is 0.5 in `home_win`; the flag keeps it visible rather than
            # letting it read as a rounding artefact (D-4).
            is_tie=(pl.col("home_win") == 0.5).cast(pl.Int8),
            split=pl.col("season").map_elements(season_split, return_dtype=pl.String),
        )
        .with_columns(
            short_week_diff=(pl.col("home_short_week") - pl.col("away_short_week")).cast(
                pl.Int8
            ),
            off_bye_diff=(pl.col("home_off_bye") - pl.col("away_off_bye")).cast(pl.Int8),
            # Indoors there is no wind to record, so zero is the observation
            # rather than an imputation - `is_dome` already tells the model
            # which it is. Outdoor games with a missing reading are a different
            # thing and get their own flag rather than being averaged over.
            # Read off `is_dome` rather than `roof` so the two agree by
            # construction - with a null roof, `~roof.is_in(...)` is itself null
            # and this flag would come through null for every future fixture.
            weather_unknown=(
                (pl.col("is_dome") == 0) & pl.col("wind").is_null()
            ).cast(pl.Int8),
            wind=pl.col("wind").fill_null(0.0).cast(pl.Float64),
        )
        .drop(
            "roof",
            "location",
            *net_cols,
            *[f"{c}_away" for c in net_cols],
        )
    )

    # Tier 3 shrinkage. The factor is the frozen constant, not a fresh
    # measurement: see TURNOVER_RELIABILITY for why it is fixed rather than
    # refit per build. `turnover_reliability` below is what produced it and is
    # what re-verifies it.
    games = games.with_columns(
        turnover_diff_shrunk=(
            pl.col(f"net_turnovers_r{_MATCHUP_WINDOW}_diff") * TURNOVER_RELIABILITY
        )
    )

    # Five 2006 games at Arrowhead have no charting data upstream, so their
    # rolling `cpoe` windows come through null (see docs/DATA_NOTES.md). Handled
    # the same way as an unknown quarterback: set to the neutral value - cpoe is
    # completion percentage *over expected*, so zero is genuinely neutral - and
    # flagged, rather than mean-imputed into invisibility.
    cpoe_nets = [c for c in net_feature_names(windows, stats) if "cpoe" in c]
    games = games.with_columns(
        cpoe_unknown=pl.any_horizontal([pl.col(c).is_null() for c in cpoe_nets]).cast(pl.Int8),
        **{c: pl.col(c).fill_null(0.0) for c in cpoe_nets},
    )

    ordered = [
        "game_id",
        "season",
        "week",
        "league_week",
        "game_type",
        "gameday",
        "split",
        "home_team",
        "away_team",
        "home_win",
        "is_tie",
        "is_postseason",
        "is_neutral_site",
        *[f"home_{c}" for c in window_cols],
        *[f"away_{c}" for c in window_cols],
        "std_is_fallback",
        *net_feature_names(windows, stats),
        *net_std_feature_names(std_stats),
        *matchup_feature_names(),
        "turnover_diff_shrunk",
        *[f"home_{c}" for c in _SIDE_COLUMNS],
        *[f"away_{c}" for c in _SIDE_COLUMNS],
        *_SIDE_DIFFS.values(),
        "elo_prob",
        *[f"{side}_{c}" for side in ("home", "away") for c in _SIDE_LABELS],
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
        "roof_unknown",
        "temp",
        "wind",
        "weather_unknown",
        "cpoe_unknown",
        "spread_line",
        "total_line",
        "home_moneyline",
        "away_moneyline",
    ]

    # 2006 onward only (D-2). The earlier seasons stay in the rolling frame so
    # 2006 week 1 has a full window behind it (D-6); they just never become rows.
    # `league_week` is ranked *after* that filter, so index 1 is 2006 week 1.
    return (
        games.filter(pl.col("season") >= MATRIX_START_SEASON)
        .with_columns(league_week=league_week_index())
        .select(ordered)
        .sort("gameday", "game_id")
    )


def to_training_arrays(
    frame: pl.DataFrame,
    features: Sequence[str] = BASELINE_FEATURES,
    target: str = "home_win",
    weight: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The sklearn boundary: polars in, ``(X, y, sample_weight)`` out.

    Ties are expanded here rather than in the matrix (D-4). The matrix says a
    tie is 0.5, which is the truth; the classifier needs a binary ``y``, so a
    tie becomes two rows at half weight - one win, one loss. Feature rows are
    duplicated to match via the index :func:`~nflpred.evaluate.expand_ties` returns.

    ``weight`` names an optional column to multiply into the sample weights,
    which is the channel Phase 7's recency weighting will use.
    """
    x_all = frame.select(features).to_numpy().astype(float)
    y_all = frame[target].to_numpy().astype(float)
    w_all = frame[weight].to_numpy().astype(float) if weight else None

    index, y, sample_weight = expand_ties(y_all, w_all)
    return x_all[index], y, sample_weight


def main() -> None:
    """Build the feature matrix from the cache and write it to processed/."""
    import argparse

    from nflpred.config import FEATURE_MATRIX_GT_PATH, TEAM_GAME_GT_PATH
    from nflpred.ingest import read_pbp, read_schedules

    parser = argparse.ArgumentParser(description="Build the game-level feature matrix.")
    parser.add_argument(
        "--garbage-time",
        action="store_true",
        help="build from the win-probability-filtered team-game table (D-9)",
    )
    args = parser.parse_args()

    team_game_path = TEAM_GAME_GT_PATH if args.garbage_time else TEAM_GAME_PATH
    matrix_path = FEATURE_MATRIX_GT_PATH if args.garbage_time else FEATURE_MATRIX_PATH
    if not team_game_path.exists():
        msg = f"{team_game_path.name} not built. Run `python -m nflpred.features.pbp_agg` first."
        raise FileNotFoundError(msg)

    team_game = pl.read_parquet(team_game_path)
    features = build_features(team_game, read_schedules(), read_pbp())
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    features.write_parquet(matrix_path, compression="zstd")

    print(
        f"game_features: {features.height:,} rows x {features.width} cols "
        f"-> {matrix_path.relative_to(matrix_path.parents[2])} "
        f"({matrix_path.stat().st_size / 1e6:.1f} MB)"
    )
    measured = turnover_reliability(team_game)
    print(
        f"turnover split-half reliability (train): measured {measured:.4f}, "
        f"applied {TURNOVER_RELIABILITY:.4f} (frozen)"
    )

    by_split = (
        features.group_by("split")
        .agg(
            games=pl.len(),
            seasons=pl.col("season").n_unique(),
            home_win_rate=pl.col("home_win").mean().round(4),
            ties=pl.col("is_tie").sum(),
        )
        .sort("seasons", descending=True)
    )
    print(by_split)

    for name, tup in (
        ("baseline", BASELINE_FEATURES),
        ("phase3", PHASE3_FEATURES),
        ("wide", wide_features()),
    ):
        nulls = {c: features[c].null_count() for c in tup if features[c].null_count() > 0}
        print(f"{name} ({len(tup)} features) with nulls: {nulls or 'none'}")


if __name__ == "__main__":
    main()
