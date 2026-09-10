import type { MetricRow } from "../api/types";

// The row labels the API's metrics tables use, from src/nflpred/evaluate.py and
// src/nflpred/api/analytics.py. One place so a rename there is a one-line change
// here rather than a hunt across pages and charts.
export const MODEL_LABEL = {
  ensemble: "ensemble",
  elo: "elo only",
  market: "market (de-vigged)",
} as const;

/** Pull the three comparator rows out of a metrics table by label. */
export function byModel(rows: MetricRow[]) {
  return {
    ens: rows.find((r) => r.model === MODEL_LABEL.ensemble),
    elo: rows.find((r) => r.model === MODEL_LABEL.elo),
    mkt: rows.find((r) => r.model === MODEL_LABEL.market),
  };
}
