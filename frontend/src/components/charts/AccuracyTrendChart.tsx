import { useMemo } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { SeasonMetric } from "../../api/types";
import { AXIS_STROKE, AXIS_TICK, GRID_STROKE } from "./ChartFrame";
import { ThemedTooltip } from "./Tooltip";

const SERIES = [
  { key: "ensemble", color: "var(--series-1)", label: "model" },
  { key: "elo", color: "var(--series-2)", label: "elo only" },
  { key: "market", color: "var(--series-3)", label: "market" },
] as const;

const NAME_IN_ROW: Record<string, string> = {
  ensemble: "ensemble",
  elo: "elo only",
  market: "market (de-vigged)",
};

// Running (cumulative) straight-up accuracy through the ordered weeks, for the
// model against its two reference lines. Cumulative rather than per-week so the
// line is readable over ~90 weeks; the shaded band is the ~66-68% range Vegas
// hits straight up, and the dashed line is the 72% "this would be a leak" mark.
export function AccuracyTrendChart({
  byWeek,
  ceiling,
  band,
}: {
  byWeek: SeasonMetric[];
  ceiling: number;
  band: [number, number];
}) {
  const data = useMemo(() => {
    const running: Record<string, { correct: number; n: number }> = {};
    for (const s of SERIES) running[s.key] = { correct: 0, n: 0 };

    return byWeek.map((wk) => {
      const point: Record<string, number | string> = {
        x: `${String(wk.season).slice(2)}w${wk.week}`,
      };
      for (const s of SERIES) {
        const row = wk.metrics.find((m) => m.model === NAME_IN_ROW[s.key]);
        if (row && row.n > 0) {
          running[s.key].correct += row.accuracy * row.n;
          running[s.key].n += row.n;
        }
        point[s.key] = running[s.key].n
          ? +(running[s.key].correct / running[s.key].n).toFixed(4)
          : NaN;
      }
      return point;
    });
  }, [byWeek]);

  return (
    <ResponsiveContainer width="100%" height={320}>
      <LineChart data={data} margin={{ top: 8, right: 20, bottom: 24, left: 4 }}>
        <CartesianGrid stroke={GRID_STROKE} strokeDasharray="2 4" />
        <ReferenceArea
          y1={band[0]}
          y2={band[1]}
          fill="var(--series-3)"
          fillOpacity={0.14}
          ifOverflow="extendDomain"
          label={{
            value: "market range",
            position: "insideBottomRight",
            fill: "var(--ink-muted)",
            fontSize: 10,
          }}
        />
        <ReferenceLine
          y={ceiling}
          stroke="var(--critical)"
          strokeDasharray="5 4"
          label={{
            value: `${Math.round(ceiling * 100)}% leakage tripwire`,
            position: "insideTopRight",
            fill: "var(--critical)",
            fontSize: 10,
          }}
        />
        <XAxis
          dataKey="x"
          tick={AXIS_TICK}
          stroke={AXIS_STROKE}
          interval="preserveStartEnd"
          minTickGap={40}
        />
        <YAxis
          domain={[0.5, 0.75]}
          ticks={[0.5, 0.55, 0.6, 0.65, 0.7, 0.75]}
          tickFormatter={(v) => `${Math.round(v * 100)}%`}
          tick={AXIS_TICK}
          stroke={AXIS_STROKE}
          width={44}
        />
        {SERIES.map((s) => (
          <Line
            key={s.key}
            type="monotone"
            dataKey={s.key}
            name={s.label}
            stroke={s.color}
            strokeWidth={2}
            dot={false}
            connectNulls
            isAnimationActive={false}
          />
        ))}
        <Legend
          verticalAlign="top"
          align="left"
          height={28}
          iconType="plainline"
          wrapperStyle={{ fontSize: 12, color: "var(--ink-2)" }}
        />
        <Tooltip
          cursor={{ stroke: "var(--axis)" }}
          content={
            <ThemedTooltip
              header={(l) => `through ${l}`}
              format={(v) => `${(v * 100).toFixed(1)}%`}
            />
          }
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
