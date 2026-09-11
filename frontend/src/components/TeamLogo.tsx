// Decorative team mark, keyed by the same team_abbr code used everywhere else
// (predictions.sqlite, GameRow.home_team/away_team). Assets live in
// frontend/public/logos/, fetched by scripts/ops/fetch_team_logos.py. The team
// code is always rendered alongside the logo, so alt is empty rather than
// double-announcing "KC" to screen readers.
export function TeamLogo({ team, size = 20 }: { team: string; size?: number }) {
  return (
    <img
      src={`/logos/${team}.png`}
      alt=""
      width={size}
      height={size}
      loading="lazy"
      className="shrink-0 object-contain"
      style={{ width: size, height: size }}
    />
  );
}
