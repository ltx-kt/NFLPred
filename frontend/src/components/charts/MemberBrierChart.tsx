import {
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { MemberBrier } from "../../api/types";
import { AXIS_STROKE, AXIS_TICK, GRID_STROKE } from "./ChartFrame";

function DotTooltip({ active, payload }: { active?: boolean; payload?: { payload: MemberBrier }[] }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="rounded-lg border border-hairline bg-surface px-3 py-2 text-xs shadow-sm">
      <div className="mb-0.5 font-medium text-ink">{d.member}</div>
      <div className="tnum text-ink-2">Brier {d.brier.toFixed(4)}</div>
      <div className="tnum text-ink-muted">
        {d.n} games · mean P {d.mean_prob.toFixed(3)}
      </div>
    </div>
  );
}

// A dot plot, not bars: the six Brier scores sit in a narrow range (~0.216 to
// ~0.224) and a bar chart would need a cut baseline to show any difference,
// which misreads as a ratio. A dot encodes position only, so the tight x-domain
// is honest. Elo, the non-learned member, is tinted apart. Lower is better.
export function MemberBrierChart({ rows }: { rows: MemberBrier[] }) {
  const data = [...rows]
    .sort((a, b) => b.brier - a.brier) // worst at top, best at bottom
    .map((r) => ({ ...r, y: r.member }));

  const values = data.map((d) => d.brier);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const pad = Math.max((hi - lo) * 0.25, 0.001);

  return (
    <ResponsiveContainer width="100%" height={Math.max(180, data.length * 38 + 48)}>
      <ScatterChart margin={{ top: 8, right: 52, bottom: 20, left: 8 }}>
        <CartesianGrid stroke={GRID_STROKE} strokeDasharray="2 4" horizontal={false} />
        <XAxis
          type="number"
          dataKey="brier"
          domain={[+(lo - pad).toFixed(4), +(hi + pad).toFixed(4)]}
          tick={AXIS_TICK}
          stroke={AXIS_STROKE}
          tickFormatter={(v) => v.toFixed(3)}
          label={{
            value: "Brier score (lower is better)",
            position: "bottom",
            fill: "var(--ink-muted)",
            fontSize: 11,
          }}
        />
        <YAxis
          type="category"
          dataKey="y"
          tick={AXIS_TICK}
          stroke={AXIS_STROKE}
          width={92}
        />
        <Scatter data={data} isAnimationActive={false} shape="circle">
          {data.map((d) => (
            <Cell
              key={d.member}
              fill={d.member === "elo" ? "var(--series-2)" : "var(--series-1)"}
            />
          ))}
        </Scatter>
        <Tooltip cursor={{ stroke: "var(--axis)", strokeDasharray: "3 3" }} content={<DotTooltip />} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}
