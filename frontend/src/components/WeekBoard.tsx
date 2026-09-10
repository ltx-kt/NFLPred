import { useWeek } from "../api/client";
import { KindBadge } from "./Badges";
import { EmptyNote, ErrorNote, Spinner } from "./Feedback";
import { GameCard } from "./GameCard";

export function WeekBoard({ season, week }: { season: number; week: number }) {
  const { data, isLoading, error } = useWeek(season, week);

  if (isLoading) return <Spinner label={`Loading ${season} week ${week}`} />;
  if (error) return <ErrorNote error={error} />;
  if (!data) return null;

  const { ref, games } = data;
  const highConf = games.filter((g) => g.confidence === "high").length;
  const decided = games.filter((g) => g.confidence !== "no read").length;

  return (
    <div>
      <div className="mb-5 flex flex-wrap items-center gap-x-3 gap-y-1">
        <h1 className="text-xl font-semibold tracking-tight">
          {season} · Week {week}
        </h1>
        <KindBadge kind={ref.kind} />
        <span className="text-sm text-ink-muted">
          {ref.n_games} {ref.n_games === 1 ? "game" : "games"} · {decided} with a lean ·{" "}
          {highConf} high confidence
          {ref.n_settled > 0 && ` · ${ref.n_settled} final`}
        </span>
      </div>

      {ref.kind === "backtest" && (
        <p className="mb-4 rounded-lg border border-hairline bg-surface-2 px-3 py-2 text-xs text-ink-2">
          This is a replayed week from a spent test season. Each pick was made by a model
          fitted only on games played before it, but the season as a whole is not an
          independent evaluation - see the About page.
        </p>
      )}

      {games.length === 0 ? (
        <EmptyNote>No predictions logged for this week.</EmptyNote>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {games.map((g) => (
            <GameCard key={g.game_id} game={g} />
          ))}
        </div>
      )}
    </div>
  );
}
