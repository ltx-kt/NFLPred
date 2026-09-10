// Small formatting helpers, shared across pages.

export const pct = (p: number | null | undefined, digits = 0): string =>
  p == null ? "-" : `${(p * 100).toFixed(digits)}%`;

export const prob3 = (p: number | null | undefined): string =>
  p == null ? "-" : p.toFixed(3);

// A prediction-log probability is P(home win). Flip it to the picked side.
export const pickProb = (homeWinProb: number, pick: string, homeTeam: string): number =>
  pick === homeTeam ? homeWinProb : 1 - homeWinProb;

export const matchup = (away: string, home: string): string => `${away} @ ${home}`;

export const shortDate = (iso: string): string => {
  const d = new Date(iso + (iso.length === 10 ? "T00:00:00" : ""));
  return Number.isNaN(d.getTime())
    ? iso
    : d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
};

// pick_correct is 1 hit / 0 miss / 0.5 tie / null. Map to a status word.
export const resultWord = (pickCorrect: number | null): "hit" | "miss" | "push" | null => {
  if (pickCorrect == null) return null;
  if (pickCorrect === 1) return "hit";
  if (pickCorrect === 0) return "miss";
  return "push";
};

export const CONFIDENCE_ORDER = ["no read", "low", "moderate", "high"] as const;
