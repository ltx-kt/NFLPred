import type { ReactNode } from "react";

// Common chrome for every chart: a titled, bordered surface with an optional
// caption. Keeps the six-checks layout consistent (title names the series when
// there is only one, caption carries the "how to read" note).
export function ChartFrame({
  title,
  caption,
  children,
}: {
  title: string;
  caption?: ReactNode;
  children: ReactNode;
}) {
  return (
    <figure className="card">
      <figcaption className="mb-3">
        <h3 className="text-sm font-semibold text-ink">{title}</h3>
        {caption && <p className="mt-0.5 text-xs text-ink-muted">{caption}</p>}
      </figcaption>
      {children}
    </figure>
  );
}

export const AXIS_TICK = { fill: "var(--ink-muted)", fontSize: 11 };
export const GRID_STROKE = "var(--grid)";
export const AXIS_STROKE = "var(--axis)";
