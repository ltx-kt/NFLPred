import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import type { GameRow } from "../api/types";
import { dateWithWeekday, matchup, pct, prob3 } from "../lib/format";
import { AgainstMarketBadge, ConfidenceBadge, ResultChip } from "./Badges";
import { ProbabilityBar } from "./ProbabilityBar";

function Anchor({
  label,
  value,
  badge,
}: {
  label: string;
  value: number | null;
  badge?: ReactNode;
}) {
  return (
    <div className="flex flex-col">
      <span className="text-[10px] uppercase tracking-wide text-ink-muted">{label}</span>
      <span className="flex items-center gap-1.5">
        <span className="tnum text-xs text-ink-2">{value == null ? "-" : prob3(value)}</span>
        {badge}
      </span>
    </div>
  );
}

export function GameCard({ game }: { game: GameRow }) {
  const settled = game.outcome != null;
  const pickHitClass =
    game.outcome?.pick_correct === 1
      ? "text-[color:var(--success-text)]"
      : game.outcome?.pick_correct === 0
        ? "text-critical"
        : "text-ink";

  // A dead-even quote (5 of 1,155 logged games) has no favored side to be
  // "against" - only flag a real disagreement.
  const marketFavorsHome = game.market_prob == null ? null : game.market_prob > 0.5;
  const marketFavorsAway = game.market_prob == null ? null : game.market_prob < 0.5;
  const againstMarket =
    (marketFavorsHome === true && game.pick !== game.home_team) ||
    (marketFavorsAway === true && game.pick !== game.away_team);

  return (
    <Link
      to={`/game/${game.game_id}`}
      className={
        "flex flex-col gap-3 card transition-colors" +
        (againstMarket
          ? " border-2 border-flag hover:border-flag"
          : " hover:border-series-1/40")
      }
    >
      <div className="flex items-start justify-between gap-2">
        <span className="text-sm font-semibold text-ink">
          {matchup(game.away_team, game.home_team)}
          <span className="ml-1.5 font-normal text-ink-muted">
            · {dateWithWeekday(game.gameday)}
          </span>
        </span>
        <div className="flex shrink-0 items-center gap-2">
          {settled && <ResultChip pickCorrect={game.outcome!.pick_correct} />}
          <ConfidenceBadge confidence={game.confidence} />
        </div>
      </div>

      <ProbabilityBar
        homeWinProb={game.home_win_prob}
        homeTeam={game.home_team}
        awayTeam={game.away_team}
      />

      <div className="flex items-baseline justify-between gap-2 text-sm">
        <span>
          <span className="text-ink-muted">pick </span>
          <span className={"font-semibold " + pickHitClass}>{game.pick}</span>
          <span className="text-ink-muted"> · {game.agreement}</span>
        </span>
        {settled && game.outcome!.home_score != null ? (
          <span className="tnum shrink-0 text-ink-2">
            {game.away_team} {game.outcome!.away_score}-{game.outcome!.home_score} {game.home_team}
          </span>
        ) : (
          <span className="shrink-0 text-xs text-ink-muted">
            {game.home_team} {pct(game.home_win_prob)}
          </span>
        )}
      </div>

      <div className="mt-auto grid grid-cols-2 gap-2 border-t border-hairline pt-2">
        <Anchor label="elo" value={game.elo_prob} />
        <Anchor
          label="market"
          value={game.market_prob}
          badge={againstMarket ? <AgainstMarketBadge /> : undefined}
        />
      </div>
    </Link>
  );
}
