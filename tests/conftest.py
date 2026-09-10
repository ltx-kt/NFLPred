"""Fixtures shared across more than one test module.

Only the ones that were byte-for-byte identical in several files live here. The
important non-member is `test_no_leakage.py`: it reads the *unfiltered* matrix on
purpose and defines its own `matrix` fixture, which overrides this one by
pytest's normal name resolution. Do not "unify" that away.
"""

from __future__ import annotations

import polars as pl
import pytest

from nflpred.config import FEATURE_MATRIX_GT_PATH
from nflpred.modeling.base import split_frame


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-skip every `data`-marked test when the feature matrix is not built.

    Individual data fixtures still `pytest.skip` with the exact build command;
    this hook just means a bare checkout does not have to reach them one by one.
    """
    if FEATURE_MATRIX_GT_PATH.exists():
        return
    skip = pytest.mark.skip(
        reason="needs data/processed/game_features_gt.parquet; "
        "run `python -m nflpred.features.build --garbage-time`"
    )
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="module")
def matrix() -> pl.DataFrame:
    """The garbage-time-filtered feature matrix - what every consumer except
    `test_no_leakage` reads. Skips, with the build command, when it is absent."""
    if not FEATURE_MATRIX_GT_PATH.exists():
        pytest.skip(
            "game_features_gt.parquet not built; "
            "run python -m nflpred.features.build --garbage-time"
        )
    return pl.read_parquet(FEATURE_MATRIX_GT_PATH)


@pytest.fixture(scope="module")
def splits(matrix: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """`(train, calib, val)` in kickoff order."""
    return tuple(split_frame(matrix, name) for name in ("train", "calib", "val"))
