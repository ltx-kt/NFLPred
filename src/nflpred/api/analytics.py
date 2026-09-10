"""Turn the log's settled-prediction frames into the metric shapes the routes return.

Every metric here comes from `nflpred.evaluate` - `accuracy`, `log_loss`, `brier`,
`metrics_table`, all tie-aware (D-4) - and every input frame from
`nflpred.predlog`. Nothing is recomputed that those modules already compute.

**On pooling config suffixes.** `predlog.config_suffix` hashes the train/calib
season *bounds*, which advance as the walk-forward does, so a 2022-2025 backtest
spans several suffixes that are all the *same recipe* - one moving train window.
Pooling those for the ensemble readout is correct and is the number
`scripts/phase7_pipeline.py` reports as the headline. What must not be pooled is
two different *recipes*: the log also holds an unweighted (`half_life = None`)
experiment over 2025 wk5-8. So the default view keeps only runs whose
``half_life`` matches the live config (:data:`nflpred.config.RECENCY_HALF_LIFE`);
``config=<suffix>`` overrides that with an exact scope. Per-member Brier is left
entirely to `predlog.rolling_brier`, which has its own newest-suffix-only rule.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import polars as pl
from sklearn.calibration import calibration_curve

from nflpred.config import RECENCY_HALF_LIFE
from nflpred.evaluate import ACCURACY_CEILING, MARKET_LABEL, expand_ties, metrics_table
from nflpred.predlog import settled_predictions

#: Vegas straight-up hit rate, for the "this is good, not suspicious" band on the
#: performance chart. The de-vigged market line in the project's own results
#: tables lands here (README).
MARKET_BAND: tuple[float, float] = (0.66, 0.68)


def _live_recipe_versions(connection: sqlite3.Connection) -> set[str]:
    """`model_version`s whose run used the live recency half-life.

    This is what makes the pooled default one recipe rather than several: every
    seeded walk-forward suffix qualifies, the unweighted 2025 experiment does not.
    """
    rows = connection.execute(
        "SELECT model_version FROM runs WHERE half_life IS ?", (RECENCY_HALF_LIFE,)
    )
    return {r["model_version"] for r in rows}


def _scoped_frame(
    connection: sqlite3.Connection, config: str | None, season: int | None
) -> tuple[pl.DataFrame, str]:
    """Settled predictions cut to one recipe (or one suffix) and optionally one season.

    Returns the frame and the label to echo back as ``config`` in the response.
    """
    frame = settled_predictions(connection)
    if frame.is_empty():
        return frame, config or "live recipe"

    if config is not None:
        frame = frame.filter(pl.col("model_version").str.ends_with(f"-{config}"))
        label = config
    else:
        keep = _live_recipe_versions(connection)
        frame = frame.filter(pl.col("model_version").is_in(list(keep)))
        label = "live recipe"

    if season is not None and not frame.is_empty():
        frame = frame.filter(pl.col("season") == season)
    return frame, label


def _comparator_entries(frame: pl.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """`(name, y_true, y_prob)` for the ensemble and its two reference lines."""
    y = frame["home_win"].to_numpy().astype(float)
    ensemble = frame["home_win_prob"].to_numpy().astype(float)
    elo = frame["elo_prob"].to_numpy().astype(float)
    market = frame["market_prob"].to_numpy().astype(float)  # NULL -> NaN, dropped downstream

    return [
        ("ensemble", y, ensemble),
        ("elo only", y, elo),
        (MARKET_LABEL, y, market),
    ]


def performance(
    connection: sqlite3.Connection, *, config: str | None, season: int | None
) -> dict:
    """Ensemble vs Elo vs market over settled games, overall and per season-week."""
    frame, label = _scoped_frame(connection, config, season)

    result: dict = {
        "config": label,
        "n_settled": frame.height,
        "overall": [],
        "by_week": [],
        "accuracy_ceiling": ACCURACY_CEILING,
        "market_band": list(MARKET_BAND),
    }
    if frame.is_empty():
        return result

    by_week = []
    for (yr, wk), group in sorted(
        frame.group_by("season", "week"), key=lambda kv: (kv[0][0], kv[0][1])
    ):
        by_week.append(
            {
                "season": int(yr),
                "week": int(wk),
                "metrics": metrics_table(_comparator_entries(group)).to_dicts(),
            }
        )

    return result | {
        "overall": metrics_table(_comparator_entries(frame)).to_dicts(),
        "by_week": by_week,
    }


def calibration(
    connection: sqlite3.Connection, *, config: str | None, season: int | None, bins: int
) -> dict:
    """Reliability points for the ensemble's calibrated probability.

    Quantile bins, tie-aware: a tied game contributes half a count to the
    home-win side and half to the home-loss side, matching
    `nflpred.evaluate.reliability_diagram`.
    """
    frame, label = _scoped_frame(connection, config, season)

    if frame.height < bins:
        return {"config": label, "bins": bins, "n": frame.height, "points": []}

    y = frame["home_win"].to_numpy().astype(float)
    p = frame["home_win_prob"].to_numpy().astype(float)
    index, labels, _ = expand_ties(y)
    observed, predicted = calibration_curve(labels, p[index], n_bins=bins, strategy="quantile")

    # Count per bin on the expanded rows, so the point sizes are honest.
    edges = np.quantile(p[index], np.linspace(0, 1, bins + 1))
    edges[-1] = np.inf
    counts = np.histogram(p[index], bins=edges)[0]

    points = [
        {"predicted": round(float(pr), 4), "observed": round(float(ob), 4), "n": int(c)}
        for pr, ob, c in zip(predicted, observed, counts, strict=False)
    ]
    return {"config": label, "bins": bins, "n": frame.height, "points": points}
