import {
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import type { CalibrationPoint } from "../../api/types";
import { AXIS_STROKE, AXIS_TICK, GRID_STROKE } from "./ChartFrame";
import { ThemedTooltip } from "./Tooltip";

// One series: the ensemble's predicted probability against the frequency it
// actually happened, in quantile bins. The dashed 45-degree line is "perfectly
// calibrated"; points above it mean the model was under-confident in that bin,
// below it over-confident. Title names the series, so no legend.
export function CalibrationChart({ points }: { points: CalibrationPoint[] }) {
  const data = points.map((p) => ({ ...p }));

  return (
    <ResponsiveContainer width="100%" height={300}>
      <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 24, left: 4 }}>
        <CartesianGrid stroke={GRID_STROKE} strokeDasharray="2 4" />
        <XAxis
          type="number"
          dataKey="predicted"
          domain={[0, 1]}
          ticks={[0, 0.25, 0.5, 0.75, 1]}
          tickFormatter={(v) => `${Math.round(v * 100)}%`}
          tick={AXIS_TICK}
          stroke={AXIS_STROKE}
          label={{
            value: "predicted P(home win)",
            position: "bottom",
            fill: "var(--ink-muted)",
            fontSize: 11,
          }}
        />
        <YAxis
          type="number"
          dataKey="observed"
          domain={[0, 1]}
          ticks={[0, 0.25, 0.5, 0.75, 1]}
          tickFormatter={(v) => `${Math.round(v * 100)}%`}
          tick={AXIS_TICK}
          stroke={AXIS_STROKE}
          width={44}
          label={{
            value: "observed frequency",
            angle: -90,
            position: "insideLeft",
            fill: "var(--ink-muted)",
            fontSize: 11,
          }}
        />
        <ZAxis dataKey="n" range={[40, 260]} name="games" />
        <ReferenceLine
          segment={[
            { x: 0, y: 0 },
            { x: 1, y: 1 },
          ]}
          stroke="var(--axis)"
          strokeDasharray="4 4"
          ifOverflow="visible"
        />
        <Line
          type="monotone"
          dataKey="observed"
          stroke="var(--series-1)"
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
        />
        <Scatter
          dataKey="observed"
          fill="var(--series-1)"
          stroke="var(--surface)"
          strokeWidth={2}
          isAnimationActive={false}
        />
        <Tooltip
          cursor={{ stroke: "var(--axis)" }}
          content={
            <ThemedTooltip
              header={(l) => `predicted ${Math.round(Number(l) * 100)}%`}
              format={(v, name) => (name === "games" ? String(v) : `${(v * 100).toFixed(1)}%`)}
            />
          }
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
