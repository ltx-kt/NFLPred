import type { MetricRow } from "../api/types";

// The row labels the API's metrics tables use, from src/nflpred/evaluate.py and
// src/nflpred/api/analytics.py. One place so a rename there is a one-line change
// here rather than a hunt across pages and charts.
export const MODEL_LABEL = {
  ensemble: "ensemble",
  elo: "elo only",
  market: "market (de-vigged)",
} as const;

// What each headline metric means, for the "?" on its stat tile. Every number
// quoted here is a mathematical reference point, not a result, so none of it
// can go stale as the log grows.
export const METRIC_HELP = {
  accuracy:
    "Share of settled games where the model's pick won. A tie counts as half a hit. It scores the pick only, not how confident the model was.",
  logLoss:
    "Scores the probability itself, not just the pick. A confident miss costs far more than a hesitant one: 90% on a loser is punished much harder than 55%. Lower is better; always saying 50/50 scores 0.693.",
  brier:
    "The mean squared gap between the predicted home-win probability and what happened (1 if the home team won, 0 if not). Like log loss but gentler on confident misses. Lower is better; always saying 50/50 scores 0.250.",
  vsMarket:
    "Model accuracy minus the betting market's accuracy on the same games, in percentage points. The market's pick is its favourite once the bookmaker's margin is removed. Positive means the model out-picked the market; the closing line is about the hardest benchmark there is, so negative is normal.",
  vsElo:
    "Model accuracy minus the accuracy of the Elo rating alone on the same games, in percentage points. Elo is the simplest member of the ensemble, so this is what the other five models and the stacking add on top of it.",
} as const;

/** Pull the three comparator rows out of a metrics table by label. */
export function byModel(rows: MetricRow[]) {
  return {
    ens: rows.find((r) => r.model === MODEL_LABEL.ensemble),
    elo: rows.find((r) => r.model === MODEL_LABEL.elo),
    mkt: rows.find((r) => r.model === MODEL_LABEL.market),
  };
}
