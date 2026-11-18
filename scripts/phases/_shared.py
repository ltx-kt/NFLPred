"""Conventions shared by the checkpoint scripts in ``scripts/phases/``.

These scripts are historical: each reproduces a metric already published in the
project's design docs. The helpers here exist so the seven of them stay
consistent - the report-path spelling, the "not built" message, and the
constraint-3 tripwire wrapper were copied by hand into every one and drifted.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nflpred.config import FEATURE_MATRIX_GT_PATH, LOGREG_MAX_ITER, REPORTS_DIR, ROOT
from nflpred.evaluate import halt_if_suspicious

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import polars as pl


def report_path(name: str) -> Path:
    """Absolute path for a generated report file, e.g. a reliability PNG.

    Just ``REPORTS_DIR / name``; the writers (`nflpred.evaluate.reliability_diagram`,
    `phase6`'s ``_importance_plot``) create the directory themselves.
    """
    return REPORTS_DIR / name


def report_written(path: Path, label: str = "reliability diagram") -> None:
    """Print the one-line "wrote it, here" notice, path relative to the repo root."""
    print(f"\n{label} -> {path.relative_to(ROOT)}")


def require_matrix(*paths: Path) -> None:
    """Raise with the build command if any feature matrix is missing.

    ``--garbage-time`` is appended for the filtered matrix, matching what each
    phase script spelled out by hand.
    """
    for path in paths:
        if not path.exists():
            flag = " --garbage-time" if path == FEATURE_MATRIX_GT_PATH else ""
            msg = f"{path.name} not built. Run `python -m nflpred.features.build{flag}`."
            raise FileNotFoundError(msg)


def checked(
    table: pl.DataFrame,
    entries: Sequence[tuple[str, np.ndarray, np.ndarray]] | None = None,
) -> pl.DataFrame:
    """Run the constraint-3 tripwire, printing the table first if it fires.

    Returns ``table`` so a caller can write ``table = checked(metrics_table(...),
    entries)``. The print-then-raise on a halt is why this is a helper: five
    phase scripts had it copied verbatim, and an eighth that forgot the print
    would halt with no visible evidence.
    """
    try:
        halt_if_suspicious(table, entries=entries)
    except SystemExit:
        print(table)
        raise
    return table


def baseline_model() -> Pipeline:
    """Scaled inputs and L2, exactly as the spec's base-estimator list says.

    L2 is left implicit: scikit-learn 1.8 deprecated the ``penalty`` keyword and
    made L2 with ``C`` the default, so passing ``penalty="l2"`` now warns while
    doing nothing. ``C=1.0`` is that L2 penalty.

    No hyperparameter search. A tuned baseline is a worse baseline: the bar is
    supposed to be what a plain model gets from its features. Phase 3 reuses it
    unchanged so its feature comparison is measured by the same model that set
    the Phase 2 bar.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=LOGREG_MAX_ITER)),
        ]
    )
