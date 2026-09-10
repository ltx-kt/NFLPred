import { useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import { useIndex, usePerformance } from "../api/client";
import { KindBadge } from "../components/Badges";
import { ErrorNote, Spinner } from "../components/Feedback";
import { StatTile } from "../components/StatTile";
import { pct, prob3, signedPct } from "../lib/format";
import { byModel } from "../lib/metrics";
import { WeekBoard } from "../components/WeekBoard";

export function History() {
  const index = useIndex();
  const [params, setParams] = useSearchParams();
  const seasons = index.data?.seasons ?? [];

  // Everything is derived from the URL, with newest-season / newest-week defaults;
  // the pickers just write ?season / ?week.
  const season = useMemo(() => {
    const p = params.get("season");
    if (p && seasons.some((s) => s.season === +p)) return +p;
    return seasons.length ? seasons[seasons.length - 1].season : undefined;
  }, [params, seasons]);

  const weeks = seasons.find((s) => s.season === season)?.weeks ?? [];
  const week = useMemo(() => {
    const p = params.get("week");
    if (p && weeks.includes(+p)) return +p;
    return weeks.length ? weeks[weeks.length - 1] : undefined;
  }, [params, weeks]);

  if (index.isLoading) return <Spinner label="Loading seasons" />;
  if (index.error) return <ErrorNote error={index.error} />;

  const kind = seasons.find((s) => s.season === season)?.kind ?? "backtest";

  return (
    <div>
      <h1 className="mb-4 text-xl font-semibold tracking-tight">History</h1>

      <div className="mb-4 flex flex-wrap items-center gap-2">
        {seasons.map((s) => (
          <button
            key={s.season}
            type="button"
            onClick={() => setParams({ season: String(s.season) })}
            className={
              "rounded-md px-2.5 py-1 text-sm ring-1 ring-inset transition-colors " +
              (s.season === season
                ? "bg-series-1/10 text-series-1 ring-series-1/30"
                : "text-ink-2 ring-hairline hover:text-ink")
            }
          >
            {s.season}
          </button>
        ))}
        <span className="ml-1">
          <KindBadge kind={kind} />
        </span>
      </div>

      {season != null && <SeasonSummary season={season} />}

      <div className="mb-5 mt-5 flex flex-wrap gap-1.5">
        {weeks.map((w) => (
          <button
            key={w}
            type="button"
            onClick={() => setParams({ season: String(season), week: String(w) })}
            className={
              "tnum h-8 w-9 rounded text-sm ring-1 ring-inset transition-colors " +
              (w === week
                ? "bg-series-1 text-white ring-series-1"
                : "text-ink-2 ring-hairline hover:text-ink")
            }
          >
            {w}
          </button>
        ))}
      </div>

      {season != null && week != null && (
        <WeekBoard key={`${season}-${week}`} season={season} week={week} />
      )}
    </div>
  );
}

function SeasonSummary({ season }: { season: number }) {
  const { data, isLoading, error } = usePerformance(undefined, season);
  if (isLoading) return <Spinner label="Scoring the season" />;
  if (error) return <ErrorNote error={error} />;
  if (!data || data.n_settled === 0)
    return <p className="text-sm text-ink-muted">No settled games in {season} yet.</p>;

  const { ens, elo, mkt } = byModel(data.overall);
  if (!ens) return null;

  const edgeVsMarket = mkt ? ens.accuracy - mkt.accuracy : null;

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      <StatTile
        label="model accuracy"
        value={pct(ens.accuracy, 1)}
        sub={`${data.n_settled} settled games`}
      />
      <StatTile label="log loss" value={prob3(ens.log_loss)} sub="lower is better" />
      <StatTile label="brier" value={prob3(ens.brier)} sub="lower is better" />
      <StatTile
        label="vs market"
        value={edgeVsMarket == null ? "-" : signedPct(edgeVsMarket)}
        sub={
          mkt
            ? `market ${pct(mkt.accuracy, 1)}${elo ? `, elo ${pct(elo.accuracy, 1)}` : ""}`
            : undefined
        }
        tone={edgeVsMarket != null && edgeVsMarket >= 0 ? "good" : "critical"}
      />
    </div>
  );
}
