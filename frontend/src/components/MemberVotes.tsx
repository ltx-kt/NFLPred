import type { MemberVote } from "../api/types";
import { pct } from "../lib/format";

const ORDER = ["logreg", "random forest", "xgboost", "lightgbm", "catboost", "elo"];

// Six members, fixed order. Each row is a mini bar for its P(home win) with a
// centre tick at 50%, so a glance shows how tightly the members agree.
export function MemberVotes({
  members,
  homeTeam,
}: {
  members: MemberVote[];
  homeTeam: string;
}) {
  const byName = new Map(members.map((m) => [m.member, m]));
  return (
    <div className="space-y-1.5">
      {ORDER.map((name) => {
        const m = byName.get(name);
        if (!m) return null;
        const home = m.pick === homeTeam;
        return (
          <div key={name} className="flex items-center gap-3 text-xs">
            <span className="w-24 shrink-0 text-ink-2">{name}</span>
            <div className="relative h-3 flex-1 overflow-hidden rounded bg-surface-2">
              <div
                className="absolute inset-y-0"
                style={{
                  left: home ? "50%" : `${m.home_win_prob * 100}%`,
                  width: `${Math.abs(m.home_win_prob - 0.5) * 100}%`,
                  background: home ? "var(--series-1)" : "var(--series-2)",
                  borderRadius: 3,
                }}
              />
              <div
                className="absolute inset-y-0"
                style={{ left: "50%", width: 1, background: "var(--axis)" }}
              />
            </div>
            <span className="tnum w-20 shrink-0 text-right text-ink-2">
              {m.pick} {pct(home ? m.home_win_prob : 1 - m.home_win_prob)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
