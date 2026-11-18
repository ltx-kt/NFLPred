"""Lint and formatting as tests, since the repo has no CI to run them.

Formatting drift had reached 37 files before this existed: nothing failed when a
file went unformatted, so nobody noticed. Running ruff here makes it a test
failure like any other.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "args", [("check", "."), ("format", "--check", ".")], ids=["lint", "format"]
)
def test_ruff_is_clean(args: tuple[str, ...]) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ruff", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
