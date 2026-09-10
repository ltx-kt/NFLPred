import { useCalibration, useIndex, useMembers, usePerformance } from "../api/client";
import { ErrorNote, Spinner } from "../components/Feedback";
import { StatTile } from "../components/StatTile";
import { AccuracyTrendChart } from "../components/charts/AccuracyTrendChart";
import { CalibrationChart } from "../components/charts/CalibrationChart";
import { ChartFrame } from "../components/charts/ChartFrame";
import { MemberBrierChart } from "../components/charts/MemberBrierChart";
import { pct, prob3, signedPct } from "../lib/format";
import { byModel } from "../lib/metrics";

export function Model() {
  const perf = usePerformance();
  const calib = useCalibration();
  const members = useMembers(4);
  const index = useIndex();

  if (perf.isLoading || calib.isLoading || members.isLoading)
    return <Spinner label="Loading diagnostics" />;
  const failed = perf.error ?? calib.error ?? members.error;
  if (failed) return <ErrorNote error={failed} />;
  if (!perf.data) return null;

  const { ens, elo, mkt } = byModel(perf.data.overall);
  const liveSeason = index.data?.live_season;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Model diagnostics</h1>
        <p className="mt-1 max-w-2xl text-sm text-ink-muted">
          Every number here is computed from settled predictions in the log. The window
          is the {perf.data.config === "live recipe" ? "walk-forward backtest" : perf.data.config}{" "}
          over {perf.data.n_settled.toLocaleString()} games - one recipe, refit each week on
          only prior games. It is not an independent test: those seasons were the model's
          test split and have been spent.
        </p>
      </div>

      {ens && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <StatTile
            label="model accuracy"
            value={pct(ens.accuracy, 1)}
            sub={mkt ? `market ${pct(mkt.accuracy, 1)}` : undefined}
          />
          <StatTile label="log loss" value={prob3(ens.log_loss)} sub="ln 2 = 0.693 is no skill" />
          <StatTile label="brier" value={prob3(ens.brier)} sub="0.25 = always 50%" />
          <StatTile
            label="vs elo baseline"
            value={elo ? signedPct(ens.accuracy - elo.accuracy) : "-"}
            sub="accuracy gain over Elo alone"
            tone={elo && ens.accuracy - elo.accuracy >= 0 ? "good" : "default"}
          />
        </div>
      )}

      <ChartFrame
        title="Calibration - ensemble"
        caption="A point on the dashed line means that when the model said X%, it happened X% of the time. Above the line = under-confident in that bin, below = over-confident."
      >
        {calib.data && calib.data.points.length > 0 ? (
          <CalibrationChart points={calib.data.points} />
        ) : (
          <p className="py-8 text-sm text-ink-muted">Not enough settled games to bin.</p>
        )}
      </ChartFrame>

      <ChartFrame
        title="Running accuracy vs the market"
        caption="Cumulative straight-up hit rate through the backtest. Shaded band is the ~66-68% range the market hits; the dashed red line is the 72% mark above which a time-split result is treated as a leak, not a model."
      >
        <AccuracyTrendChart
          byWeek={perf.data.by_week}
          ceiling={perf.data.accuracy_ceiling}
          band={perf.data.market_band}
        />
      </ChartFrame>

      <ChartFrame
        title="Per-member Brier score"
        caption={
          members.data?.note ??
          "Each ensemble member scored on its own probability. Monitoring only - never fed back into the weights."
        }
      >
        {members.data && members.data.overall.length > 0 ? (
          <MemberBrierChart rows={members.data.overall} />
        ) : (
          <p className="py-8 text-sm text-ink-muted">
            No settled games for the current config
            {liveSeason ? ` (${liveSeason} has not been played)` : ""}.
          </p>
        )}
      </ChartFrame>
    </div>
  );
}
