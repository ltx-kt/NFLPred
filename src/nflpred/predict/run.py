"""Refit on everything before kickoff, predict the week, and log it."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nflpred._provenance import short_hash
from nflpred.backtest import WeeklyFit, fit_week
from nflpred.config import (
    CALIB_WINDOW_SEASONS,
    RECENCY_HALF_LIFE,
)
from nflpred.evaluate import market_probability
from nflpred.modeling.base import home_win_probability
from nflpred.modeling.ensemble import ENSEMBLE_FEATURES, meta_coefficients, predict_records
from nflpred.predict.week import WeekView
from nflpred.predlog import (
    config_suffix,
    connect,
    library_versions,
    model_version,
    record_predictions,
    record_run,
)

# ------------------------------------------------------------- prediction


def frame_market_probability(frame: pl.DataFrame) -> np.ndarray:
    """De-vigged home win probability from each row's moneylines."""
    return market_probability(
        frame["home_moneyline"].to_numpy(), frame["away_moneyline"].to_numpy()
    )


def _matrix_fingerprint(frame: pl.DataFrame) -> str:
    """Content hash of the rows a fit actually saw.

    `nflpred.modeling.store` hashes the matrix *file*; there is no file here, because
    the live path rebuilds in memory to admit an unplayed row. Hashing the
    frame's contents answers the same question the manifest's hash answers:
    were these predictions made from the data we think they were.
    """
    return short_hash(frame.hash_rows().to_numpy().tobytes())


def predict_week(
    view: WeekView,
    half_life: float | None = RECENCY_HALF_LIFE,
    calib_seasons: int = CALIB_WINDOW_SEASONS,
    explain: bool = False,
) -> tuple[list[dict[str, Any]], WeeklyFit, np.ndarray]:
    """Refit on everything before kickoff, then predict the week.

    The fit is kept in memory and returned rather than saved and reloaded.
    D-30 leaves native model formats deferred for exactly this reason: there is
    nothing to reload *to*. ``--explain`` needs the raw inner estimators and the
    stack's ``final_estimator_``, and `nflpred.modeling.store.load_models` could not
    supply them anyway - its fingerprint recompute requires a matrix that still
    contains the 2019-2021 validation rows, which a live 2026 matrix does not
    single out.
    """
    fit = fit_week(
        view.completed,
        view.league_week,
        view.season,
        view.week,
        ENSEMBLE_FEATURES,
        half_life,
        calib_seasons,
    )

    scored = view.target.with_columns(market_prob=pl.Series(frame_market_probability(view.target)))
    probability = home_win_probability(fit.stack, scored, ENSEMBLE_FEATURES)
    records = predict_records(fit.stack, fit.members, scored)

    if explain:
        # The Phase 6 call sequence, unchanged: the explainers are built over
        # the **raw** inner estimators, which are the objects the meta-learner
        # consumed and therefore the objects to explain.
        from nflpred.explain import explained_records
        from nflpred.explain.attribution import build_explainers, meta_terms

        coefficients, intercept = meta_terms(fit.stack)
        explainers = build_explainers(fit.raw, float(fit.train["elo_prob"].mean()))
        records = explained_records(
            records, scored, explainers, coefficients, intercept, probability
        )

    return records, fit, probability


def log_week(
    view: WeekView,
    fit: WeeklyFit,
    records: Sequence[dict[str, Any]],
    path: Path | None = None,
) -> str:
    """Write one week's run and predictions, and return the model version."""
    versions = library_versions()
    train_seasons, calib_seasons = fit.seasons("train"), fit.seasons("calib")
    suffix = config_suffix(train_seasons, calib_seasons, fit.half_life, ENSEMBLE_FEATURES, versions)
    version = model_version(view.season, view.week, suffix)

    connection = connect(path) if path else connect()
    try:
        record_run(
            connection,
            version,
            season=view.season,
            week=view.week,
            league_week=view.league_week,
            train_seasons=train_seasons,
            n_train=fit.train.height,
            calib_seasons=calib_seasons,
            n_calib=fit.calib.height,
            half_life=fit.half_life,
            features=ENSEMBLE_FEATURES,
            matrix_sha256_16=_matrix_fingerprint(view.completed),
            versions=versions,
            meta_coefficients=meta_coefficients(fit.stack),
        )
        record_predictions(
            connection,
            version,
            records,
            view.target,
            ENSEMBLE_FEATURES,
            market=frame_market_probability(view.target),
        )
    finally:
        connection.close()
    return version
