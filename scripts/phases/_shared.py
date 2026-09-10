"""Conventions shared by the checkpoint scripts in ``scripts/phases/``.

These scripts are historical: each reproduces a metric already published in
``docs/RESULTS.md``. The report-path spelling and the "wrote -> path" line are
here so the seven of them stay consistent and so the printed path is relative to
the repo root rather than to wherever the reports tree happens to sit.
"""

from __future__ import annotations

from pathlib import Path

from nflpred.config import REPORTS_DIR, ROOT


def report_path(name: str) -> Path:
    """Absolute path for a generated report file, e.g. a reliability PNG.

    Just ``REPORTS_DIR / name``; the writers (`nflpred.evaluate.reliability_diagram`,
    `phase6`'s ``_importance_plot``) create the directory themselves.
    """
    return REPORTS_DIR / name


def report_written(path: Path, label: str = "reliability diagram") -> None:
    """Print the one-line "wrote it, here" notice, path relative to the repo root."""
    print(f"\n{label} -> {path.relative_to(ROOT)}")
