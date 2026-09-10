"""Metrics, comparators, and reliability diagrams.

Constraint 4 says accuracy is the headline but not the target: every checkpoint
reports accuracy, log loss and Brier together, against the same comparators, so
numbers from different phases are comparable. Written once here rather than
re-derived per phase.

Two things this module is careful about.

**Ties.** The target is a float in {0, 0.5, 1} (D-4). Rather than special-casing
each metric, a tie row is expanded into two half-weight rows — ``y=1`` and
``y=0`` — and every metric is computed on the weighted expansion. Under log loss
and Brier that is exactly right, and it gives a tie half credit on accuracy for
free. :func:`expand_ties` is also what the training path uses, so the model and
the scoreboard agree on what a tie is.

**Comparators.** Always-pick-home is the floor; the de-vigged market line is the
reference ceiling (~66-68% straight up). The market number is context for how
much signal is left on the table, not a target to chase.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss as _sk_log_loss

#: Probabilities are clipped this far from {0, 1} before log loss, so a single
#: confident miss cannot return infinity and hide every other number.
_EPS: float = 1e-15

#: Vegas hits ~66-68% straight up. Anything above this on a proper time split is
#: a leak, not a model (constraint 3). Lives here rather than in a phase script
#: because two of them now enforce it, and a tripwire duplicated per phase is a
#: tripwire that eventually differs per phase.
ACCURACY_CEILING: float = 0.72

#: Comparators are exempt from the tripwire: the market line is allowed to be
#: good, and a leak is not what it would mean if it were.
_CEILING_EXEMPT: frozenset[str] = frozenset({"market (de-vigged)"})


def expand_ties(
    y: np.ndarray, weight: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand {0, 0.5, 1} labels into binary labels with sample weights (D-4).

    Returns ``(row_index, y_binary, sample_weight)``. A decided game yields one
    row at its original weight; a tie yields two rows at half weight each, one
    labelled a home win and one a home loss. ``row_index`` maps every emitted
    row back to its position in the input, so a caller can index the matching
    feature rows or predicted probabilities with it.

    Weights compose by multiplication, which is what lets Phase 7's recency
    weighting ride the same channel.
    """
    y = np.asarray(y, dtype=float)
    base = np.ones_like(y) if weight is None else np.asarray(weight, dtype=float)

    decided = y != 0.5
    tied = ~decided

    index = np.concatenate([np.flatnonzero(decided), np.flatnonzero(tied), np.flatnonzero(tied)])
    labels = np.concatenate([y[decided], np.ones(tied.sum()), np.zeros(tied.sum())])
    weights = np.concatenate([base[decided], base[tied] * 0.5, base[tied] * 0.5])

    return index, labels, weights


def accuracy(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    """Straight-up hit rate, with half credit for a tie."""
    index, labels, weights = expand_ties(y_true)
    predicted = (np.asarray(y_prob, dtype=float)[index] >= threshold).astype(float)
    return float(np.average(predicted == labels, weights=weights))


def log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Binary cross-entropy. An uninformative model scores ln(2) ≈ 0.693."""
    index, labels, weights = expand_ties(y_true)
    probs = np.clip(np.asarray(y_prob, dtype=float)[index], _EPS, 1 - _EPS)
    return float(_sk_log_loss(labels, probs, labels=[0, 1], sample_weight=weights))


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared probability error. Always-0.5 scores 0.25."""
    index, labels, weights = expand_ties(y_true)
    probs = np.asarray(y_prob, dtype=float)[index]
    return float(brier_score_loss(labels, probs, sample_weight=weights))


def metrics_table(entries: Sequence[tuple[str, np.ndarray, np.ndarray]]) -> pl.DataFrame:
    """Score several models on the same games and return one small table.

    ``entries`` is ``(name, y_true, y_prob)`` per model. Rows where the
    probability is null are dropped per model and reported in ``n``, so a
    comparator with partial coverage (the market line, before 2007) is still
    scored honestly on the games it covers.
    """
    rows = []
    for name, y_true, y_prob in entries:
        y_arr = np.asarray(y_true, dtype=float)
        p_arr = np.asarray(y_prob, dtype=float)
        keep = ~np.isnan(p_arr)
        y_arr, p_arr = y_arr[keep], p_arr[keep]

        rows.append(
            {
                "model": name,
                "n": int(keep.sum()),
                "accuracy": accuracy(y_arr, p_arr),
                "log_loss": log_loss(y_arr, p_arr),
                "brier": brier(y_arr, p_arr),
            }
        )

    return pl.DataFrame(rows).with_columns(
        pl.col("accuracy").round(4),
        pl.col("log_loss").round(4),
        pl.col("brier").round(4),
    )


def _exact_accuracy(
    entries: Sequence[tuple[str, np.ndarray, np.ndarray]],
) -> dict[str, float]:
    """Per-model accuracy, unrounded, with the same null-drop as `metrics_table`."""
    out: dict[str, float] = {}
    for name, y_true, y_prob in entries:
        p_arr = np.asarray(y_prob, dtype=float)
        keep = ~np.isnan(p_arr)
        out[name] = accuracy(np.asarray(y_true, dtype=float)[keep], p_arr[keep])
    return out


def halt_if_suspicious(
    table: pl.DataFrame,
    ceiling: float = ACCURACY_CEILING,
    exempt: frozenset[str] = _CEILING_EXEMPT,
    *,
    entries: Sequence[tuple[str, np.ndarray, np.ndarray]] | None = None,
) -> None:
    """Raise if any model in ``table`` scored above the accuracy ceiling.

    Constraint 3's tripwire, executable. It raises rather than warns because the
    failure mode it guards against is a leak that reads as a triumph — and the
    one thing that must not happen is that number being reported. The caller
    should print the table first, so the evidence is visible above the traceback.

    ``entries`` is the same ``(name, y_true, y_prob)`` list handed to
    :func:`metrics_table`. When given, the check runs on the *unrounded*
    accuracy: ``metrics_table`` rounds to four places for display, so a genuine
    0.72004 shows as 0.7200 and a ``pl.col("accuracy") > 0.72`` test would let it
    through — the tripwire rounding in the leak's favour. Without ``entries`` the
    check falls back to the rounded ``table["accuracy"]`` column, which is enough
    for a hand-built table in a test.
    """
    if entries is not None:
        exact = _exact_accuracy(entries)
        over = [
            name
            for name in table["model"].to_list()
            if name not in exempt and exact.get(name, 0.0) > ceiling
        ]
        suspicious = table.filter(pl.col("model").is_in(over))
    else:
        suspicious = table.filter(
            (pl.col("accuracy") > ceiling) & ~pl.col("model").is_in(list(exempt))
        )
    if not suspicious.height:
        return

    names = ", ".join(suspicious["model"].to_list())
    msg = (
        f"Halted: {names} scored above {ceiling:.0%} accuracy on a time-ordered "
        f"split. Vegas hits 66-68%. This is a leak, not a result — investigate "
        f"before reporting anything above."
    )
    raise SystemExit(msg)


# ------------------------------------------------------------- comparators


def always_home(n: int, home_win_rate: float) -> np.ndarray:
    """Constant-probability home pick.

    Picking the home team every week is the trivial baseline. Emitting the
    *rate* rather than 1.0 keeps its log loss finite, and since the constant is
    above 0.5 the accuracy is identical to a hard always-home rule. The rate
    must come from training seasons only — reading it off the evaluation set
    would be a (small) leak.
    """
    return np.full(n, float(home_win_rate))


def implied_probability(moneyline: np.ndarray) -> np.ndarray:
    """American moneyline to vig-inclusive implied probability."""
    ml = np.asarray(moneyline, dtype=float)
    # `where` evaluates both branches, so the unused one divides by zero at
    # ml = ±100 and by NaN wherever a line is missing. Both are discarded.
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ml > 0, 100.0 / (ml + 100.0), -ml / (-ml + 100.0))


def market_probability(home_ml: np.ndarray, away_ml: np.ndarray) -> np.ndarray:
    """De-vigged home win probability from the two moneylines.

    Proportional (normalised) de-vigging: the two implied probabilities sum to
    slightly more than 1, and the excess is removed pro rata. Cruder than
    Shin's method, and close enough — this is a reference line, not a model.
    Games without a quoted line come back as NaN and are dropped by
    :func:`metrics_table`.
    """
    home = implied_probability(home_ml)
    away = implied_probability(away_ml)
    total = home + away
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, home / total, np.nan)


# -------------------------------------------------------------- diagnostics


def reliability_diagram(
    entries: Sequence[tuple[str, np.ndarray, np.ndarray]],
    path: Path,
    n_bins: int = 10,
    title: str = "Reliability",
) -> Path:
    """Plot predicted vs observed frequency and write a PNG.

    Ties are expanded unweighted here (``calibration_curve`` has no weight
    channel), which double-counts a dozen games across two decades and changes
    nothing visible. Constraint 3 asks for this plot for every model, so the
    signature takes a list.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1, label="perfect")

    for name, y_true, y_prob in entries:
        p_arr = np.asarray(y_prob, dtype=float)
        keep = ~np.isnan(p_arr)
        index, labels, _ = expand_ties(np.asarray(y_true, dtype=float)[keep])
        observed, predicted = calibration_curve(
            labels, p_arr[keep][index], n_bins=n_bins, strategy="quantile"
        )
        ax.plot(predicted, observed, marker="o", markersize=4, linewidth=1.2, label=name)

    ax.set_xlabel("predicted probability of a home win")
    ax.set_ylabel("observed frequency")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
