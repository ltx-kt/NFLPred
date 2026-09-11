import type { WeekKind } from "../api/types";
import { resultWord } from "../lib/format";

// "backtest" vs "live" - the single most important label on the whole dashboard.
// 2022-2025 is the spent test split; a replay of it is not an independent result.
export function KindBadge({ kind }: { kind: WeekKind }) {
  const live = kind === "live";
  return (
    <span
      className={
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium uppercase tracking-wide " +
        (live
          ? "bg-series-1/10 text-series-1"
          : "bg-surface-2 text-ink-muted ring-1 ring-inset ring-hairline")
      }
      title={
        live
          ? "Live season - these games have not been played."
          : "Backtest - a completed season replayed by a model fitted only on earlier games. Not an independent evaluation."
      }
    >
      {live ? "live" : "backtest"}
    </span>
  );
}

const CONF_STYLE: Record<string, string> = {
  "no read": "bg-surface-2 text-ink-muted ring-hairline",
  low: "bg-surface-2 text-ink-2 ring-hairline",
  moderate: "bg-series-1/10 text-series-1 ring-series-1/20",
  high: "bg-series-1/15 text-series-1 ring-series-1/30",
};

export function ConfidenceBadge({ confidence }: { confidence: string }) {
  return (
    <span
      className={
        "inline-flex rounded px-1.5 py-0.5 text-[11px] font-medium ring-1 ring-inset " +
        (CONF_STYLE[confidence] ?? CONF_STYLE.low)
      }
      title="Confidence band from the calibrated probability's distance from 50%, fixed a priori (not fitted)."
    >
      {confidence}
    </span>
  );
}

// Flags a pick that differs from the market favorite. A caution flag, not a
// trophy: historically these hit 42% vs 69% when the model agrees with the
// market - see the tooltip.
export function VsMarketBadge() {
  return (
    <span
      className="inline-flex rounded px-1.5 py-0.5 text-[11px] font-medium ring-1 ring-inset bg-flag/10 text-flag ring-flag/25"
      title="This pick differs from the market favorite. Historically, picks against the market have hit 42% of the time, vs 69% when the model agrees with it."
    >
      vs market
    </span>
  );
}

export function ResultChip({ pickCorrect }: { pickCorrect: number | null }) {
  const word = resultWord(pickCorrect);
  if (!word) return null;
  const style =
    word === "hit"
      ? "bg-good/12 text-[color:var(--success-text)] ring-good/25"
      : word === "miss"
        ? "bg-critical/10 text-critical ring-critical/25"
        : "bg-surface-2 text-ink-muted ring-hairline";
  return (
    <span
      className={"inline-flex rounded px-1.5 py-0.5 text-[11px] font-semibold uppercase ring-1 ring-inset " + style}
    >
      {word}
    </span>
  );
}
