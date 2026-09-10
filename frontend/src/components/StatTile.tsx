// A single headline number. No plot, so per the dataviz skill it needs no hover.

export function StatTile({
  label,
  value,
  sub,
  tone = "default",
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: "default" | "good" | "critical";
}) {
  const valueColor =
    tone === "good"
      ? "text-[color:var(--success-text)]"
      : tone === "critical"
        ? "text-critical"
        : "text-ink";
  return (
    <div className="card">
      <div className="text-[11px] uppercase tracking-wide text-ink-muted">{label}</div>
      <div className={"tnum mt-1 text-2xl font-semibold " + valueColor}>{value}</div>
      {sub && <div className="mt-0.5 text-xs text-ink-muted">{sub}</div>}
    </div>
  );
}
