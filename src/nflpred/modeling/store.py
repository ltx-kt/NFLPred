"""Writing fitted models to disk, and reading them back with the receipts.

Artifacts here are **outputs, not a cache** (D-17). A phase script always
refits and always overwrites; nothing skips a fit because a pickle exists. A
full refit of all ten models takes about fifteen seconds, which is cheaper than
the class of bug where a stale artifact silently answers for a model definition
that has since changed.

What makes that checkable is a **prediction fingerprint**: the first 32
calibrated probabilities on the validation split, rounded, recorded in the
manifest at save time and recomputed at load time. A model that comes back
predicting something else is not the model that was saved, whatever the file
name says, and :func:`load_models` raises rather than returning it.

``root`` is a parameter throughout so that pointing this at object storage later
is a swap rather than a rewrite. Native per-library formats (``Booster.save_model``
and friends) are deliberately deferred to Phase 7, where cross-version
portability starts to matter; joblib is right for artifacts read by the same
environment that wrote them.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import polars as pl
import sklearn
from sklearn.calibration import CalibratedClassifierCV

from nflpred.config import MODELS_DIR
from nflpred.modeling.base import home_win_probability, split_frame

#: How many validation probabilities the fingerprint covers, and to how many
#: places. 32 games is plenty to catch a different model and short enough to
#: read in a diff; 6 places is well inside float64 round-trip precision but
#: outside the last-bit noise a library patch release can move.
FINGERPRINT_N: Final[int] = 32
FINGERPRINT_PLACES: Final[int] = 6

MANIFEST_NAME: Final[str] = "manifest.json"


def _slug(name: str) -> str:
    """Model name to file stem. ``random forest`` -> ``random_forest``."""
    return name.replace(" ", "_")


def fingerprint(
    model: CalibratedClassifierCV, matrix: pl.DataFrame, features: Sequence[str]
) -> list[float]:
    """The first :data:`FINGERPRINT_N` calibrated probabilities on validation.

    Validation rather than train, because a model that was refitted on shifted
    data will differ there too, and validation is the split every Phase 4 number
    is quoted on. This reads no test row.
    """
    validation = split_frame(matrix, "val").head(FINGERPRINT_N)
    probabilities = home_win_probability(model, validation, features)
    return [round(float(p), FINGERPRINT_PLACES) for p in probabilities]


def _matrix_hash(path: Path) -> str:
    """Content hash of the feature matrix the models were fitted from."""
    return sha256(path.read_bytes()).hexdigest()[:16]


def _versions() -> dict[str, str]:
    """Library versions that can move a prediction. Recorded, not enforced —
    a mismatch is context for a failed fingerprint, not a failure by itself."""
    import catboost
    import lightgbm
    import xgboost

    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "lightgbm": lightgbm.__version__,
        "catboost": catboost.__version__,
    }


def save_models(
    models: Mapping[str, CalibratedClassifierCV],
    key: str,
    matrix: pl.DataFrame,
    features: Sequence[str],
    meta: Mapping[str, Any] | None = None,
    matrix_path: Path | None = None,
    root: Path = MODELS_DIR,
) -> Path:
    """Write ``models`` under ``root/key`` with a manifest, and return that path.

    ``key`` names the run — the checkpoint uses the feature set, so
    ``models/core16/`` and ``models/phase3_29/`` sit side by side rather than
    overwriting each other. Anything already there is replaced: these are
    outputs of the current definitions, and a directory holding two generations
    of model would be exactly the ambiguity D-17 rules out.
    """
    directory = root / key
    directory.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        joblib.dump(model, directory / f"{_slug(name)}.joblib")

    manifest = {
        "key": key,
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "features": list(features),
        "n_features": len(features),
        "models": sorted(models),
        "matrix_path": str(matrix_path) if matrix_path else None,
        "matrix_sha256_16": _matrix_hash(matrix_path) if matrix_path else None,
        "splits": {
            name: split_frame(matrix, name).height for name in ("train", "calib", "val")
        },
        "versions": _versions(),
        "fingerprint": {
            name: fingerprint(model, matrix, features) for name, model in models.items()
        },
        **(meta or {}),
    }
    (directory / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return directory


def load_models(
    key: str, matrix: pl.DataFrame, root: Path = MODELS_DIR
) -> tuple[dict[str, CalibratedClassifierCV], dict[str, Any]]:
    """Load a saved run and verify it still predicts what it did when saved.

    Returns ``(models, manifest)``. Raises if a file or the manifest is missing,
    or if any model's recomputed fingerprint disagrees with the recorded one —
    loudly, because a silently-wrong model is the failure mode that would
    quietly poison every number downstream of it.

    ``matrix`` must be the matrix the models were fitted from; the manifest
    records its hash so a caller can check that before trusting a mismatch
    report.
    """
    directory = root / key
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.exists():
        msg = f"no manifest at {manifest_path}. Refit and save — artifacts are outputs (D-17)."
        raise FileNotFoundError(msg)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    features = manifest["features"]

    models: dict[str, CalibratedClassifierCV] = {}
    mismatched: list[str] = []

    for name in manifest["models"]:
        path = directory / f"{_slug(name)}.joblib"
        if not path.exists():
            msg = f"{manifest_path} lists '{name}' but {path.name} is missing."
            raise FileNotFoundError(msg)

        model = joblib.load(path)
        expected = manifest["fingerprint"][name]
        actual = fingerprint(model, matrix, features)
        if not np.allclose(actual, expected, atol=10.0**-FINGERPRINT_PLACES):
            mismatched.append(
                f"  {name}: manifest {expected[:3]}... vs loaded {actual[:3]}..."
            )
        models[name] = model

    if mismatched:
        msg = (
            f"prediction fingerprint mismatch in {directory} — the loaded models do "
            f"not predict what was saved:\n" + "\n".join(mismatched) + "\n"
            f"Either the feature matrix changed under them (manifest hash "
            f"{manifest.get('matrix_sha256_16')}) or the artifacts are stale. "
            f"Refit rather than trusting these."
        )
        raise ValueError(msg)

    return models, manifest
