"""Download NFL team logos into frontend/public/logos/, one PNG per team_abbr.

One-off: the frontend has no logo assets, so the game cards need a local,
offline-safe set keyed by the same team codes already in the prediction log
(`nflreadpy.load_teams()["team_abbr"]` - verified to cover every code seen in
`data/predictions.sqlite`, including the LA/WAS/JAX cases that differ between
feeds). Bundled rather than hotlinked so the Docker image and dev/offline use
don't depend on an external host at view time.

Re-run to refresh the set (e.g. after an expansion/relocation); it always
overwrites.

Run::

    uv run python scripts/ops/fetch_team_logos.py
"""

from __future__ import annotations

from pathlib import Path

import httpx
import nflreadpy as nfl

OUT_DIR = Path(__file__).resolve().parents[2] / "frontend" / "public" / "logos"


def main() -> None:
    teams = nfl.load_teams()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for abbr, url in zip(
            teams["team_abbr"].to_list(), teams["team_logo_espn"].to_list(), strict=True
        ):
            resp = client.get(url)
            resp.raise_for_status()
            (OUT_DIR / f"{abbr}.png").write_bytes(resp.content)
            print(f"{abbr}: {len(resp.content)} bytes")

    print(f"wrote {teams.height} logos to {OUT_DIR}")


if __name__ == "__main__":
    main()
