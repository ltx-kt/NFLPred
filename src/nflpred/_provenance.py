"""Provenance helpers: the short content hash and the library versions.

:func:`short_hash` is the one spelling of "truncated sha256" behind every
recorded fingerprint - the ingest column hash, the matrix hashes, the config
suffix and the per-game feature hash. Changing it orphans every hash already on
disk or in the log.

`nflpred.modeling.store` records this in a model manifest and `nflpred.predlog`
records it in a `runs` row, for the same reason: recorded, not enforced. A
mismatch is context for a number that moved, not a failure on its own.

`predlog.config_suffix` hashes this dict, so the **key names must not change** -
a new or renamed key orphans every `model_version` already in the log.
"""

from __future__ import annotations

import platform
from hashlib import sha256

#: Hex characters kept from a sha256 digest - 64 bits, ample for collision-free
#: identity at this project's scale.
SHORT_HASH_LENGTH = 16


def short_hash(data: bytes | str, length: int = SHORT_HASH_LENGTH) -> str:
    """Leading ``length`` hex characters of the sha256 of ``data`` (str is UTF-8)."""
    raw = data.encode() if isinstance(data, str) else data
    return sha256(raw).hexdigest()[:length]


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
