import type { ReactNode } from "react";

// Common chrome for every chart: a titled, bordered surface with an optional
// caption. Keeps the six-checks layout consistent (title names the series when
// there is only one, caption carries the "how to read" note).
export function ChartFrame({
  title,
  caption,
  children,
  right,
}: {
  title: string;
  caption?: ReactNode;
  children: ReactNode;
  right?: ReactNode;
}) {
  return (
    <figure className="rounded-xl border border-hairline bg-surface p-4">
      <figcaption className="mb-3 flex items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-ink">{title}</h3>
          {caption && <p className="mt-0.5 text-xs text-ink-muted">{caption}</p>}
        </div>
        {right}
      </figcaption>
      {children}
    </figure>
  );
}

export const AXIS_TICK = { fill: "var(--ink-muted)", fontSize: 11 };
export const GRID_STROKE = "var(--grid)";
export const AXIS_STROKE = "var(--axis)";
