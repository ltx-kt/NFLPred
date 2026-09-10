import type { TooltipProps } from "recharts";

// Themed replacement for Recharts' default white tooltip. `rows` maps each
// payload entry to a label/value pair; pass `header` for the x-axis context.
export function ThemedTooltip({
  active,
  payload,
  label,
  header,
  format,
}: TooltipProps<number, string> & {
  header?: (label: string | number | undefined) => string;
  format?: (value: number, name: string) => string;
}) {
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div className="rounded-lg border border-hairline bg-surface px-3 py-2 text-xs shadow-sm">
      {header && <div className="mb-1 font-medium text-ink">{header(label)}</div>}
      <ul className="space-y-0.5">
        {payload.map((entry) => (
          <li key={String(entry.name)} className="flex items-center gap-2">
            <span
              className="h-2 w-2 rounded-sm"
              style={{ background: entry.color }}
              aria-hidden
            />
            <span className="text-ink-2">{entry.name}</span>
            <span className="tnum ml-auto font-medium text-ink">
              {typeof entry.value === "number"
                ? (format?.(entry.value, String(entry.name)) ?? entry.value.toFixed(3))
                : entry.value}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
