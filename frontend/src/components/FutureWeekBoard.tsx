import type { ScheduleGame } from "../api/types";
import { KindBadge } from "./Badges";
import { FutureGameCard } from "./FutureGameCard";

// The History-page counterpart to WeekBoard, for a week with no `predictions`
// row yet - fed the slice of an already-fetched schedule response rather than
// making its own request, since History already holds the full season.
export function FutureWeekBoard({
  season,
  week,
  games,
}: {
  season: number;
  week: number;
  games: ScheduleGame[];
}) {
  const predicted = games.filter((g) => g.pick != null).length;

  return (
    <div>
      <div className="mb-5 flex flex-wrap items-center gap-x-3 gap-y-1">
        <h1 className="text-xl font-semibold tracking-tight">
          {season} · Week {week}
        </h1>
        <KindBadge kind="live" />
        <span className="text-sm text-ink-muted">
          {games.length} {games.length === 1 ? "game" : "games"}
          {predicted > 0 && ` · ${predicted} predicted`}
        </span>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {games.map((g) => (
          <FutureGameCard key={g.game_id} game={g} />
        ))}
      </div>
    </div>
  );
}
