"""nflreadpy loaders with a manifest-keyed parquet cache.

Downloading is the expensive step (~372 columns x ~47k rows x 27 seasons), so
every pull is column-selected and written to ``data/raw`` as parquet. Reruns hit
the cache.

The cache key is a manifest entry, not merely the file's existence. It records
the hash of the requested column list, so widening ``PBP_COLUMNS`` in a later
phase invalidates the affected files instead of silently serving a narrower
frame that fails much further downstream.

Freshness rule: a season below :data:`~nflpred.config.LIVE_SEASON` is complete and
therefore immutable — never re-fetched once cached. The live season and the
schedule file are re-fetched once they age past their TTL.

Run directly to populate the cache::

    python -m nflpred.ingest                      # all seasons in INGEST_SEASONS
    python -m nflpred.ingest --seasons 2020:2025
    python -m nflpred.ingest --force              # ignore the cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import nflreadpy as nfl
import polars as pl
from nflreadpy.config import update_config

from nflpred.config import (
    DATA_RAW,
    INGEST_SEASONS,
    LIVE_SEASON,
    MANIFEST_PATH,
    PBP_COLUMNS,
    PBP_DIR,
    SCHEDULE_COLUMNS,
    SCHEDULES_PATH,
    TEAM_ALIASES,
    TEAM_STATS_DIR,
)

#: How stale a mutable file may get before we re-fetch it.
LIVE_TTL_SECONDS: int = 12 * 60 * 60

_NFLREADPY_VERSION: str = version("nflreadpy")

# nflreadpy caches whole 372-column frames in RAM by default. We select columns
# and persist parquet ourselves, so its cache is pure overhead here.
update_config(cache_mode="off", verbose=False)


# ----------------------------------------------------------------- manifest


def _columns_hash(columns: Sequence[str]) -> str:
    """Stable short hash of a column selection, order-insensitive."""
    joined = "\n".join(sorted(columns))
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def load_manifest() -> dict[str, Any]:
    """Read the cache manifest, or an empty one if it does not exist yet."""
    if not MANIFEST_PATH.exists():
        return {}
    with MANIFEST_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def save_manifest(manifest: dict[str, Any]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)


def _key(path: Path) -> str:
    """Manifest key: path relative to ``data/raw``, forward-slashed."""
    return path.relative_to(DATA_RAW).as_posix()


def _is_immutable(season: int | None) -> bool:
    """True when a season is complete and can never change again."""
    return season is not None and season < LIVE_SEASON


def _is_cached(path: Path, columns: Sequence[str], season: int | None) -> bool:
    """Decide whether the on-disk file can be served without re-fetching."""
    if not path.exists():
        return False

    entry = load_manifest().get(_key(path))
    if entry is None:
        return False

    # A changed column selection must invalidate, or later phases silently read
    # a frame missing the columns they asked for.
    if entry.get("columns_hash") != _columns_hash(columns):
        return False

    if _is_immutable(season):
        return True

    fetched = datetime.fromisoformat(entry["fetched_at"])
    return (datetime.now(UTC) - fetched).total_seconds() < LIVE_TTL_SECONDS


def _record(
    manifest: dict[str, Any],
    path: Path,
    *,
    source: str,
    season: int | None,
    columns: Sequence[str],
    frame: pl.DataFrame,
) -> None:
    manifest[_key(path)] = {
        "source": source,
        "season": season,
        "columns_hash": _columns_hash(columns),
        "n_columns": frame.width,
        "n_rows": frame.height,
        "nflreadpy_version": _NFLREADPY_VERSION,
        "fetched_at": datetime.now(UTC).isoformat(),
        "bytes": path.stat().st_size,
        "immutable": _is_immutable(season),
    }


def _write(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path, compression="zstd")


def _select(frame: pl.DataFrame, columns: Sequence[str], label: str) -> pl.DataFrame:
    """Select the requested columns, failing loudly on any that vanished."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        msg = (
            f"{label}: nflreadpy {_NFLREADPY_VERSION} no longer provides "
            f"{missing}. Update the selection in src/nflpred/config.py."
        )
        raise KeyError(msg)
    return frame.select(columns)


# ------------------------------------------------------------------- paths


def pbp_path(season: int) -> Path:
    return PBP_DIR / f"pbp_{season}.parquet"


def team_stats_path(season: int) -> Path:
    return TEAM_STATS_DIR / f"team_stats_{season}.parquet"


# ------------------------------------------------------------------ ingest


def _load_season(
    load: Callable[[], pl.DataFrame], season: int, label: str
) -> pl.DataFrame | None:
    """Pull one season, returning ``None`` when the live one is not published.

    Before week 1 the live season has a schedule and nothing else — nflverse
    either serves an empty frame or 404s, depending on the file. Neither is an
    error: it is the ordinary August state, and the only season it can happen
    for is the one being predicted. Any other season failing is a real failure
    and is raised.

    An empty frame is still written and recorded, so the 12-hour TTL governs the
    retry rather than every run re-reading a file that upstream has published but
    not yet filled. A failed *fetch* records nothing and is retried next run,
    which costs one 404 and keeps the manifest free of entries for files that do
    not exist.
    """
    try:
        frame = load()
    except Exception as error:  # noqa: BLE001 — re-raised below unless live
        if season < LIVE_SEASON:
            raise
        print(f"  {label} {season}: not published yet ({type(error).__name__}); skipped")
        return None

    if not frame.height and season >= LIVE_SEASON:
        print(f"  {label} {season}: published but empty; the season has not started")
    return frame


def ingest_pbp(seasons: Iterable[int], *, force: bool = False) -> None:
    """Cache column-selected play-by-play, one parquet file per season."""
    manifest = load_manifest()

    for season in seasons:
        path = pbp_path(season)
        if not force and _is_cached(path, PBP_COLUMNS, season):
            print(f"  pbp {season}: cached")
            continue

        started = time.perf_counter()
        frame = _load_season(
            lambda year=season: _select(
                nfl.load_pbp([year]), PBP_COLUMNS, f"load_pbp({year})"
            ),
            season,
            "pbp",
        )
        if frame is None:
            continue
        _write(frame, path)
        _record(
            manifest,
            path,
            source="load_pbp",
            season=season,
            columns=PBP_COLUMNS,
            frame=frame,
        )
        print(
            f"  pbp {season}: {frame.height:>6,} rows x {frame.width} cols  "
            f"{path.stat().st_size / 1e6:>5.1f} MB  ({time.perf_counter() - started:.1f}s)"
        )

    save_manifest(manifest)


def ingest_team_stats(seasons: Iterable[int], *, force: bool = False) -> None:
    """Cache weekly team stats.

    Not a feature source — this is nflverse's own team-game aggregation, kept
    so Phase 1 can reconcile our play-by-play rollup against an independent
    implementation.
    """
    manifest = load_manifest()

    for season in seasons:
        path = team_stats_path(season)
        if not force and _is_cached(path, ("*",), season):
            print(f"  team_stats {season}: cached")
            continue

        frame = _load_season(
            lambda year=season: nfl.load_team_stats([year], summary_level="week"),
            season,
            "team_stats",
        )
        if frame is None:
            continue
        _write(frame, path)
        _record(
            manifest,
            path,
            source="load_team_stats",
            season=season,
            columns=("*",),
            frame=frame,
        )
        print(f"  team_stats {season}: {frame.height:>5,} rows x {frame.width} cols")

    save_manifest(manifest)


def ingest_schedules(*, force: bool = False) -> None:
    """Cache all schedules in one file.

    Always treated as mutable: it carries future games whose scores, lines, and
    kickoff times are still being filled in.
    """
    manifest = load_manifest()

    if not force and _is_cached(SCHEDULES_PATH, SCHEDULE_COLUMNS, None):
        print("  schedules: cached")
        return

    frame = _select(nfl.load_schedules(), SCHEDULE_COLUMNS, "load_schedules()")
    _write(frame, SCHEDULES_PATH)
    _record(
        manifest,
        SCHEDULES_PATH,
        source="load_schedules",
        season=None,
        columns=SCHEDULE_COLUMNS,
        frame=frame,
    )
    print(
        f"  schedules: {frame.height:,} rows x {frame.width} cols  "
        f"(seasons {frame['season'].min()}-{frame['season'].max()})"
    )

    save_manifest(manifest)


def ingest_all(seasons: Iterable[int] = INGEST_SEASONS, *, force: bool = False) -> None:
    seasons = list(seasons)
    print(f"Ingesting {len(seasons)} seasons ({seasons[0]}-{seasons[-1]})")
    started = time.perf_counter()

    ingest_schedules(force=force)
    ingest_pbp(seasons, force=force)
    ingest_team_stats(seasons, force=force)

    total = sum(p.stat().st_size for p in DATA_RAW.rglob("*.parquet"))
    print(
        f"Done in {time.perf_counter() - started:.1f}s. "
        f"Cache: {total / 1e6:.1f} MB across {len(list(DATA_RAW.rglob('*.parquet')))} files."
    )


# -------------------------------------------------------------------- read


def normalize_teams(frame: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
    """Canonicalise team abbreviations and turn blanks into nulls.

    Applied on read rather than on write, so the parquet cache stays a faithful
    copy of upstream and every consumer still sees one vocabulary. See
    :data:`~nflpred.config.TEAM_ALIASES` for why the current codes win.
    """
    present = [c for c in columns if c in frame.columns]
    return frame.with_columns(
        [
            pl.col(c)
            .replace(TEAM_ALIASES)
            # pbp carries "" as a stand-in for "no possession" in 1999-2000
            .replace({"": None})
            .alias(c)
            for c in present
        ]
    )


def _cached_paths(
    seasons: Iterable[int], to_path: Callable[[int], Path], label: str
) -> list[Path]:
    """Cache files for ``seasons``, tolerating an absent **live** season.

    A missing completed season is a broken cache and raises. A missing live
    season is the ordinary state of affairs between February and September:
    nflverse publishes the schedule long before the first snap, and the live
    season's play-by-play file does not exist — or exists and is empty — until
    week 1 kicks off. Warning and skipping is what lets
    ``python -m nflpred.predict --season 2026 --week 1`` work in August, which is
    the whole point of extending :data:`~nflpred.config.INGEST_SEASONS` to it.
    """
    keep: list[Path] = []
    missing: list[str] = []

    for season in seasons:
        path = to_path(season)
        if path.exists() and pl.scan_parquet(path).select(pl.len()).collect().item():
            keep.append(path)
        elif season >= LIVE_SEASON:
            warnings.warn(
                f"{label} {season}: no rows cached yet — the live season has not "
                f"started. Skipping it; every feature it needs comes from prior "
                f"seasons and the schedule.",
                stacklevel=3,
            )
        else:
            missing.append(path.name)

    if missing:
        msg = f"Not cached: {missing}. Run `python -m nflpred.ingest` first."
        raise FileNotFoundError(msg)
    return keep


def read_pbp(seasons: Iterable[int] = INGEST_SEASONS) -> pl.DataFrame:
    """Read cached play-by-play. Raises if a completed season is not ingested."""
    paths = _cached_paths(seasons, pbp_path, "pbp")
    frame = pl.concat([pl.read_parquet(p) for p in paths], how="vertical")
    return normalize_teams(frame, ("posteam", "defteam", "home_team", "away_team"))


def read_schedules() -> pl.DataFrame:
    if not SCHEDULES_PATH.exists():
        msg = "schedules.parquet not cached. Run `python -m nflpred.ingest` first."
        raise FileNotFoundError(msg)
    frame = pl.read_parquet(SCHEDULES_PATH)
    return normalize_teams(frame, ("home_team", "away_team"))


def read_team_stats(seasons: Iterable[int] = INGEST_SEASONS) -> pl.DataFrame:
    paths = _cached_paths(seasons, team_stats_path, "team_stats")
    frame = pl.concat([pl.read_parquet(p) for p in paths], how="diagonal")
    return normalize_teams(frame, ("team", "opponent_team"))


# --------------------------------------------------------------------- cli


def _parse_seasons(spec: str) -> list[int]:
    """Parse ``1999:2025`` or ``2020,2021,2022`` into a season list."""
    if ":" in spec:
        lo, hi = (int(x) for x in spec.split(":", 1))
        return list(range(lo, hi + 1))
    return [int(x) for x in spec.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache nflverse data locally.")
    parser.add_argument(
        "--seasons",
        type=_parse_seasons,
        default=list(INGEST_SEASONS),
        help="e.g. 1999:2025 or 2023,2024 (default: all ingest seasons)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-fetch even when the manifest reports a valid cache entry",
    )
    args = parser.parse_args()
    ingest_all(args.seasons, force=args.force)


if __name__ == "__main__":
    main()
