import { useIndex } from "../api/client";
import { EmptyNote, ErrorNote, Spinner } from "../components/Feedback";
import { WeekBoard } from "../components/WeekBoard";

export function ThisWeek() {
  const index = useIndex();

  if (index.isLoading) return <Spinner label="Finding the latest week" />;
  if (index.error) return <ErrorNote error={index.error} />;

  const latest = index.data?.latest;
  if (!latest)
    return (
      <EmptyNote>
        The prediction log is empty. Run <code>uv run python -m nflpred.predict --season 2026 --week 1</code>{" "}
        or the seed script, then reload.
      </EmptyNote>
    );

  return <WeekBoard season={latest.season} week={latest.week} />;
}
