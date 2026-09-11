import { pct } from "../lib/format";
import { TeamLogo } from "./TeamLogo";

// P(home win) as a split bar: home fill grows from the right, away from the left,
// a 2px surface gap between them, a hairline at the 50% mark. The favored side is
// whichever crosses the midpoint - that side is filled with --prob-fav (the
// model's pick), the other with --prob-dog. Team codes and logos anchor each end
// so identity is never carried by position alone.
export function ProbabilityBar({
  homeWinProb,
  homeTeam,
  awayTeam,
  height = 10,
}: {
  homeWinProb: number;
  homeTeam: string;
  awayTeam: string;
  height?: number;
}) {
  const home = Math.round(homeWinProb * 100);
  const away = 100 - home;
  const homeFav = homeWinProb >= 0.5;
  const homeColor = homeFav ? "var(--prob-fav)" : "var(--prob-dog)";
  const awayColor = homeFav ? "var(--prob-dog)" : "var(--prob-fav)";

  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between">
        <TeamLogo team={awayTeam} size={36} />
        <TeamLogo team={homeTeam} size={36} />
      </div>
      <div className="mb-1 flex items-baseline justify-between gap-2 text-xs">
        <span
          className={
            "tnum whitespace-nowrap " + (homeFav ? "text-ink-muted" : "font-semibold text-ink")
          }
        >
          {awayTeam} {pct(1 - homeWinProb)}
        </span>
        <span
          className={
            "tnum whitespace-nowrap " + (homeFav ? "font-semibold text-ink" : "text-ink-muted")
          }
        >
          {pct(homeWinProb)} {homeTeam}
        </span>
      </div>
      <div
        className="relative flex w-full overflow-hidden rounded"
        style={{ height, background: "var(--surface-2)" }}
        role="img"
        aria-label={`${awayTeam} ${away} percent, ${homeTeam} ${home} percent`}
      >
        <div style={{ width: `${away}%`, background: awayColor, borderRadius: 4 }} />
        <div style={{ width: 2, background: "var(--surface)" }} />
        <div style={{ width: `calc(${home}% - 2px)`, background: homeColor, borderRadius: 4 }} />
        <div
          className="absolute inset-y-0"
          style={{ left: "50%", width: 1, background: "var(--axis)" }}
          aria-hidden
        />
      </div>
    </div>
  );
}
