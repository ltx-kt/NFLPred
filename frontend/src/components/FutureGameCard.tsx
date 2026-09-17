import type { ScheduleGame } from "../api/types";
import { dateWithWeekday, matchup, pct } from "../lib/format";
import { ConfidenceBadge } from "./Badges";
import { Anchor } from "./GameCard";
import { ProbabilityBar } from "./ProbabilityBar";
import { TeamLogo } from "./TeamLogo";

// Same logo/label/bar geometry as ProbabilityBar, with nothing to plot yet -
// keeps a FutureGameCard the same height as a GameCard in the same grid slot.
function BlankProbabilityBar({ homeTeam, awayTeam }: { homeTeam: string; awayTeam: string }) {
  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between">
        <TeamLogo team={awayTeam} size={36} />
        <TeamLogo team={homeTeam} size={36} />
      </div>
      <div className="mb-1 flex items-baseline justify-between gap-2 text-xs">
        <span className="whitespace-nowrap text-ink-muted">{awayTeam}</span>
        <span className="whitespace-nowrap text-ink-muted">{homeTeam}</span>
      </div>
      <div
        className="w-full rounded"
        style={{ height: 10, background: "var(--surface-2)" }}
        aria-hidden
      />
    </div>
  );
}

// A week the log has no `predictions` row for yet - only a `schedule` row
// (matchup, date; sometimes a pick, if the week has been predicted but not
// yet played). No `game_id` here resolves on /game/:id, so this never links
// out the way GameCard does. Structurally mirrors GameCard section-for-section
// (header / probability / pick row / anchors) so the two card kinds are the
// same size wherever a page might show either.
export function FutureGameCard({ game }: { game: ScheduleGame }) {
  const predicted = game.pick != null && game.home_win_prob != null;
  // The picked team's own probability, not the raw (home-team) one - showing
  // "pick GB 24%" when home_win_prob is NYJ's would read as a mismatch.
  const pickProb = predicted
    ? game.pick === game.home_team
      ? game.home_win_prob!
      : 1 - game.home_win_prob!
    : null;

  return (
    <div className="flex flex-col gap-3 card">
      <div className="flex items-start justify-between gap-2">
        <span className="text-sm font-semibold text-ink">
          {matchup(game.away_team, game.home_team)}
          <span className="ml-1.5 font-normal text-ink-muted">
            · {game.gameday ? dateWithWeekday(game.gameday) : "TBD"}
          </span>
        </span>
        <div className="flex shrink-0 items-center gap-2">
          {predicted && game.confidence && <ConfidenceBadge confidence={game.confidence} />}
        </div>
      </div>

      {predicted ? (
        <ProbabilityBar
          homeWinProb={game.home_win_prob!}
          homeTeam={game.home_team}
          awayTeam={game.away_team}
        />
      ) : (
        <BlankProbabilityBar homeTeam={game.home_team} awayTeam={game.away_team} />
      )}

      <div className="flex items-baseline justify-between gap-2 text-sm">
        {predicted ? (
          <span>
            <span className="text-ink-muted">pick </span>
            <span className="font-semibold text-ink">{game.pick}</span>
          </span>
        ) : (
          <span className="text-ink-muted">not yet predicted</span>
        )}
        {predicted && (
          <span className="tnum shrink-0 text-xs text-ink-muted">
            {game.pick} {pct(pickProb)}
          </span>
        )}
      </div>

      <div className="mt-auto grid grid-cols-2 gap-2 border-t border-hairline pt-2">
        <Anchor label="elo" value={null} />
        <Anchor label="market" value={null} />
      </div>
    </div>
  );
}
