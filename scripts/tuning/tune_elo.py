"""Elo hyperparameter grid, scored on the training seasons only (D-11).

The spec left K, home-field advantage and the margin-of-victory multiplier
unspecified. This script resolves them the only way that keeps the later
comparison honest: grid them against **2006-2015** - the same seasons every
phase trains on (D-5) - and freeze the winner into :mod:`nflpred.config` as a
constant.

Two rules make this auditable rather than a moving target:

* **Validation sees no tuning decision.** 2016-2018 and 2019-2021 are not
  scored here, and 2022-2025 is not opened. A hyperparameter chosen on the
  validation set would make every Phase 3 number a selection artefact.
* **The grid is not re-searched.** Its output is quoted in a comment beside the
  constants it produced. This script exists so that comment can be reproduced,
  not so the constants can drift.

Ratings still burn in from 1999 - only the *scoring* window is restricted. A
grid run on ratings that started cold in 2006 would be choosing a K for a
different rating system than the one Phase 3 uses.

Writes nothing. Run::

    uv run python scripts/tune_elo.py
"""

from __future__ import annotations

import polars as pl

from nflpred.config import ELO_HFA, ELO_K, ELO_MOV, TRAIN_SEASONS
from nflpred.evaluate import accuracy, brier, log_loss
from nflpred.features.elo import build_elo
from nflpred.ingest import read_schedules

#: The grid. Small on purpose - four K values spanning the range anyone
#: sensible would pick, three plausible home-field advantages, and the MOV
#: multiplier as the one structural choice. 24 cells, ~7,000 games each.
K_GRID: tuple[float, ...] = (12.0, 16.0, 20.0, 24.0)
HFA_GRID: tuple[float, ...] = (45.0, 55.0, 65.0)
MOV_GRID: tuple[bool, ...] = (True, False)

pl.Config.set_tbl_rows(30)
pl.Config.set_tbl_width_chars(120)


def _score(schedules: pl.DataFrame, k: float, hfa: float, mov: bool) -> dict[str, float]:
    """Log loss, Brier and accuracy of the Elo-only probability on train seasons."""
    elo = build_elo(schedules, k=k, hfa=hfa, mov=mov)

    # Home rows only: one prediction per game, not two mirrored ones.
    train = elo.filter(
        (pl.col("is_home") == 1)
        & pl.col("season").is_between(*TRAIN_SEASONS)
    ).join(
        schedules.select("game_id", "home_score", "away_score"),
        on="game_id",
        how="inner",
    ).filter(pl.col("home_score").is_not_null())

    margin = train["home_score"] - train["away_score"]
    outcome = (
        pl.when(margin > 0).then(1.0).when(margin < 0).then(0.0).otherwise(0.5)
    )
    y = train.select(outcome.alias("y"))["y"].to_numpy()
    p = train["elo_prob"].to_numpy()

    return {
        "k": k,
        "hfa": hfa,
        "mov": mov,
        "n": train.height,
        "log_loss": log_loss(y, p),
        "brier": brier(y, p),
        "accuracy": accuracy(y, p),
    }


def main() -> None:
    schedules = read_schedules()

    rows = [
        _score(schedules, k, hfa, mov)
        for mov in MOV_GRID
        for k in K_GRID
        for hfa in HFA_GRID
    ]

    grid = pl.DataFrame(rows).with_columns(
        pl.col("log_loss").round(4),
        pl.col("brier").round(4),
        pl.col("accuracy").round(4),
    ).sort("log_loss")

    print(
        f"Elo grid - scored on {TRAIN_SEASONS[0]}-{TRAIN_SEASONS[1]} only "
        f"({rows[0]['n']:,} games). Ratings burn in from 1999.\n"
    )
    print(grid)

    best = grid.row(0, named=True)
    print(
        f"\nbest: K={best['k']:.0f}  HFA={best['hfa']:.0f}  MOV={best['mov']}  "
        f"-> log loss {best['log_loss']:.4f}"
    )
    print(
        f"frozen in src/nflpred/config.py: ELO_K={ELO_K:.0f}  ELO_HFA={ELO_HFA:.0f}  "
        f"ELO_MOV={ELO_MOV}"
    )
    if (best["k"], best["hfa"], best["mov"]) != (ELO_K, ELO_HFA, ELO_MOV):
        print(
            "\nThe frozen constants no longer match this grid's winner. Update them "
            "deliberately, or leave them and record why - do not let them drift."
        )


if __name__ == "__main__":
    main()
