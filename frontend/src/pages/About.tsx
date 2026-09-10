export function About() {
  return (
    <div className="prose-sm max-w-2xl space-y-5 text-sm leading-relaxed text-ink-2">
      <h1 className="text-xl font-semibold tracking-tight text-ink">How to read this</h1>

      <section>
        <h2 className="font-semibold text-ink">What the number is</h2>
        <p>
          Every game gets a probability that the home team wins, from an ensemble of six
          models (five gradient-boosted / linear learners and an Elo rating), combined and
          then <em>calibrated</em>. Calibrated means the number is meant to be taken
          literally: across all the games where it said 65%, the home team should win about
          65% of the time. The Model page shows how close it actually gets.
        </p>
      </section>

      <section>
        <h2 className="font-semibold text-ink">Confidence bands</h2>
        <p>
          The band comes only from how far the probability sits from 50%, fixed in advance,
          not fitted:
        </p>
        <ul className="ml-4 list-disc space-y-0.5">
          <li>
            <strong>no read</strong> - within 3 points of a coin flip; the model is
            declining to have an opinion
          </li>
          <li>
            <strong>low</strong> - 3 to 7 points
          </li>
          <li>
            <strong>moderate</strong> - 7 to 13 points
          </li>
          <li>
            <strong>high</strong> - more than 13 points from 50%
          </li>
        </ul>
      </section>

      <section>
        <h2 className="font-semibold text-ink">Why ~65% is good</h2>
        <p>
          NFL games are close to coin flips. Picking the home team every week gets you into
          the high 50s. The betting market, with the vig removed, lands around 66-68%
          straight up - that is roughly the ceiling. This model sits just under it. Anything
          above 72% on a proper time-ordered split is treated as a bug (a data leak that
          reads as a triumph), not a breakthrough.
        </p>
      </section>

      <section>
        <h2 className="font-semibold text-ink">Backtest vs live</h2>
        <p>
          Weeks tagged <strong>backtest</strong> are completed seasons replayed: each pick
          was made by a model trained only on games played before it, so the picks are
          honest, but those seasons were the model's test split and have now been spent.
          Read them as a sanity check on the method, not as an independent result. Weeks
          tagged <strong>live</strong> are the current season, predicted before kickoff.
        </p>
      </section>

      <section>
        <h2 className="font-semibold text-ink">What is not here</h2>
        <p>
          No point spreads, no bet sizing, no injury feed. The projected starting
          quarterback is whoever started a team's previous game, so a mid-week change is not
          reflected until the following week. Weather is only known for outdoor stadiums.
        </p>
      </section>
    </div>
  );
}
