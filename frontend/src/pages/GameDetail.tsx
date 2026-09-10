import { useNavigate, useParams } from "react-router-dom";
import { useGame } from "../api/client";
import type { ExplanationFactor } from "../api/types";
import { ConfidenceBadge, ResultChip } from "../components/Badges";
import { ErrorNote, Spinner } from "../components/Feedback";
import { MemberVotes } from "../components/MemberVotes";
import { ProbabilityBar } from "../components/ProbabilityBar";
import { matchup, prob3, shortDate } from "../lib/format";

export function GameDetail() {
  const { gameId } = useParams();
  const navigate = useNavigate();
  const { data, isLoading, error } = useGame(gameId);

  if (isLoading) return <Spinner label="Loading game" />;
  if (error) return <ErrorNote error={error} />;
  if (!data) return null;

  const expl = data.record.explanation;
  const outcome = data.outcome;

  return (
    <div className="space-y-6">
      <button
        type="button"
        onClick={() => navigate(-1)}
        className="text-xs text-ink-muted hover:text-ink"
      >
        &larr; back
      </button>

      <header>
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <h1 className="text-xl font-semibold tracking-tight">
            {matchup(data.away_team, data.home_team)}
          </h1>
          <ConfidenceBadge confidence={data.confidence} />
          {outcome && <ResultChip pickCorrect={outcome.pick_correct} />}
        </div>
        <p className="mt-1 text-sm text-ink-muted">
          {data.season} week {data.week} · {shortDate(data.gameday)} ·{" "}
          <span className="tnum">{data.model_version}</span>
        </p>
      </header>

      <section className="card">
        <ProbabilityBar
          homeWinProb={data.home_win_prob}
          homeTeam={data.home_team}
          awayTeam={data.away_team}
          height={14}
        />
        <div className="mt-3 flex flex-wrap items-center gap-x-6 gap-y-1 text-sm">
          <span>
            <span className="text-ink-muted">pick </span>
            <span className="font-semibold">{data.pick}</span>
          </span>
          <span className="text-ink-2">{data.agreement} members agree</span>
          <span className="tnum text-ink-2">elo {prob3(data.elo_prob)}</span>
          <span className="tnum text-ink-2">market {prob3(data.market_prob)}</span>
          {outcome && outcome.home_score != null && (
            <span className="tnum text-ink-2">
              final {data.away_team} {outcome.away_score} - {outcome.home_score} {data.home_team}
            </span>
          )}
        </div>
      </section>

      <section>
        <h2 className="mb-2 text-sm font-semibold">Member votes</h2>
        <div className="card">
          <MemberVotes members={data.members} homeTeam={data.home_team} />
        </div>
      </section>

      {expl ? (
        <>
          <section>
            <h2 className="mb-2 text-sm font-semibold">Why</h2>
            <p className="card text-sm leading-relaxed text-ink-2">
              {expl.narrative}
            </p>
          </section>

          <section className="grid gap-4 sm:grid-cols-2">
            <FactorList title="Points toward the pick" factors={expl.top_factors_for} sign="for" />
            <FactorList
              title="Points against it"
              factors={expl.top_factors_against}
              sign="against"
            />
          </section>

          <section>
            <h2 className="mb-2 text-sm font-semibold">Confidence drivers</h2>
            <dl className="divide-y divide-hairline rounded-xl border border-hairline bg-surface text-sm">
              {Object.entries(expl.confidence_drivers).map(([k, v]) => (
                <div key={k} className="px-4 py-2.5">
                  <dt className="text-xs uppercase tracking-wide text-ink-muted">
                    {k.replace(/_/g, " ")}
                  </dt>
                  <dd className="text-ink-2">{v}</dd>
                </div>
              ))}
            </dl>
            {expl.dissent && <p className="mt-2 text-xs text-ink-muted">{expl.dissent}</p>}
          </section>
        </>
      ) : (
        <p className="rounded-xl border border-hairline bg-surface-2 px-4 py-3 text-sm text-ink-muted">
          No written explanation for this game - it was logged as part of a backtest, which
          skips the SHAP attribution to keep the replay fast. Live weekly runs include it.
        </p>
      )}

      {data.available_versions.length > 1 && (
        <p className="text-xs text-ink-muted">
          Logged under {data.available_versions.length} configs:{" "}
          <span className="tnum">{data.available_versions.join(", ")}</span>. Showing the
          newest.
        </p>
      )}
    </div>
  );
}

function FactorList({
  title,
  factors,
  sign,
}: {
  title: string;
  factors: ExplanationFactor[];
  sign: "for" | "against";
}) {
  return (
    <div className="card">
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-ink-muted">
        {title}
      </h3>
      {factors.length === 0 ? (
        <p className="text-sm text-ink-muted">None - the model had no read here.</p>
      ) : (
        <ul className="space-y-2">
          {factors.map((f) => (
            <li key={f.feature} className="flex items-start gap-2 text-sm">
              <span
                className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full"
                style={{
                  background: sign === "for" ? "var(--series-1)" : "var(--series-2)",
                }}
                aria-hidden
              />
              <span className="text-ink-2">
                {f.plain}
                <span className="tnum ml-1 text-xs text-ink-muted">
                  ({f.shap >= 0 ? "+" : ""}
                  {f.shap.toFixed(3)})
                </span>
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
