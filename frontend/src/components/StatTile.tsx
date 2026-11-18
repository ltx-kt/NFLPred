import type { ReactNode } from "react";
import { InfoTip } from "./InfoTip";

// A single headline number. No plot, so per the dataviz skill it needs no data
// hover; `help` adds a "?" that explains what the number means.

export function StatTile({
  label,
  value,
  sub,
  tone = "default",
  help,
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: "default" | "good" | "critical";
  help?: ReactNode;
}) {
  const valueColor =
    tone === "good"
      ? "text-[color:var(--success-text)]"
      : tone === "critical"
        ? "text-critical"
        : "text-ink";
  return (
    <div className="card">
      <div className="flex items-center justify-between gap-2">
        <div className="text-[11px] uppercase tracking-wide text-ink-muted">{label}</div>
        {help && <InfoTip label={label}>{help}</InfoTip>}
      </div>
      <div className={"tnum mt-1 text-2xl font-semibold " + valueColor}>{value}</div>
      {sub && <div className="mt-0.5 text-xs text-ink-muted">{sub}</div>}
    </div>
  );
}
