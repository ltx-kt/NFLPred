"""Library versions that can move a prediction.

`nflpred.modeling.store` records this in a model manifest and `nflpred.predlog`
records it in a `runs` row, for the same reason: recorded, not enforced. A
mismatch is context for a number that moved, not a failure on its own.

`predlog.config_suffix` hashes this dict, so the **key names must not change** -
a new or renamed key orphans every `model_version` already in the log.
"""

from __future__ import annotations

import platform


def library_versions() -> dict[str, str]:
    """Python plus the four fitting libraries whose version can move a fit."""
    import catboost
    import lightgbm
    import sklearn
    import xgboost

    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "lightgbm": lightgbm.__version__,
        "catboost": catboost.__version__,
    }
