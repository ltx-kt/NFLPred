// Hand-written mirror of src/api/schemas.py. Diff the two in review; there is no
// codegen step for seven endpoints.

export type WeekKind = "live" | "backtest";

export interface Health {
  runs: number;
  predictions: number;
  member_predictions: number;
  outcomes: number;
  settled: number;
  settled_share: number;
  model_versions: string[];
}

export interface WeekRef {
  season: number;
  week: number;
  kind: WeekKind;
  n_games: number;
  n_settled: number;
}

export interface SeasonIndex {
  season: number;
  kind: WeekKind;
  weeks: number[];
}

export interface Index {
  seasons: SeasonIndex[];
  latest: WeekRef | null;
  config_suffixes: string[];
  live_season: number;
}

export interface MemberVote {
  member: string;
  home_win_prob: number;
  pick: string;
}

export interface Outcome {
  home_score: number | null;
  away_score: number | null;
  home_win: number; // 1.0 home win, 0.0 away win, 0.5 tie (D-4)
  pick_correct: number | null; // 1.0 hit, 0.0 miss, 0.5 tie, null if no-read
}

export interface GameRow {
  game_id: string;
  model_version: string;
  config: string;
  generated_at: string;
  season: number;
  week: number;
  gameday: string;
  home_team: string;
  away_team: string;
  pick: string;
  home_win_prob: number;
  confidence: string;
  agreement: string;
  elo_prob: number | null;
  market_prob: number | null;
  members: MemberVote[];
  outcome: Outcome | null;
  has_explanation: boolean;
}

export interface Week {
  ref: WeekRef;
  games: GameRow[];
}

// The output-contract record from nflpred.modeling.ensemble.predict_records, with an
// optional Phase 6 explanation block. Passed through untouched by the API, so
// this type is intentionally loose.
export interface EnsembleRecord {
  game_id: string;
  home_team: string;
  away_team: string;
  model_votes: Record<string, { pick: string; home_win_prob: number }>;
  ensemble: { pick: string; home_win_prob: number; confidence: string };
  agreement: string;
  explanation?: Explanation;
}

export interface ExplanationFactor {
  feature: string;
  value: number;
  shap: number;
  plain: string;
}

export interface Explanation {
  top_factors_for: ExplanationFactor[];
  top_factors_against: ExplanationFactor[];
  confidence_drivers: Record<string, string>;
  dissent: string;
  vs_baselines: { elo_only: number | null; market_implied: number | null };
  narrative: string;
  [k: string]: unknown;
}

export interface GameDetail {
  game_id: string;
  model_version: string;
  config: string;
  season: number;
  week: number;
  gameday: string;
  home_team: string;
  away_team: string;
  pick: string;
  home_win_prob: number;
  confidence: string;
  agreement: string;
  elo_prob: number | null;
  market_prob: number | null;
  members: MemberVote[];
  outcome: Outcome | null;
  record: EnsembleRecord;
  available_versions: string[];
}

export interface MetricRow {
  model: string;
  n: number;
  accuracy: number;
  log_loss: number;
  brier: number;
}

export interface SeasonMetric {
  season: number;
  week: number | null;
  metrics: MetricRow[];
}

export interface Performance {
  config: string;
  n_settled: number;
  overall: MetricRow[];
  by_week: SeasonMetric[];
  accuracy_ceiling: number;
  market_band: [number, number];
}

export interface CalibrationPoint {
  predicted: number;
  observed: number;
  n: number;
}

export interface Calibration {
  config: string;
  bins: number;
  n: number;
  points: CalibrationPoint[];
}

export interface MemberBrier {
  member: string;
  n: number;
  brier: number;
  mean_prob: number;
}

export interface Members {
  config: string;
  window: number;
  overall: MemberBrier[];
  trailing: MemberBrier[];
  note: string;
}
