"""Static half of the leak-hygiene claim, for every checkpoint script.

Each phase script reads its split by name. If ``"test"`` is not a string
constant anywhere in the module, and ``TEST_SEASONS`` is not imported, then no
code path in it can select a held-out row. The runtime half - poison the held-out
rows and show nothing downstream moves - lives in `test_models.py`,
`test_ensemble.py` and `test_explanations.py`.

Two scripts are **excluded on purpose**: phase 7 spends the test split (D-1) and
phase 1 prints the split boundaries in its summary, so both name ``TEST_SEASONS``
by design and neither selects a row with it. Phases 2 and 3 had no static check
before this file; they get one for free here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_PHASES = Path(__file__).resolve().parents[1] / "scripts" / "phases"

#: The checkpoint scripts that must never name the held-out split, plus the
#: module they share - logic that moved into `_shared.py` still has to pass.
HYGIENIC_SCRIPTS = [
    _PHASES / "_shared.py",
    _PHASES / "phase2_baseline.py",
    _PHASES / "phase3_features.py",
    _PHASES / "phase4_models.py",
    _PHASES / "phase5_ensemble.py",
    _PHASES / "phase6_explanations.py",
]


@pytest.mark.parametrize("script", HYGIENIC_SCRIPTS, ids=lambda p: p.stem)
def test_script_never_names_the_held_out_split(script: Path) -> None:
    tree = ast.parse(script.read_text(encoding="utf-8"))

    constants = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "test" not in constants, f"{script.name} names the test split"

    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "TEST_SEASONS" not in imported, f"{script.name} imports TEST_SEASONS"
