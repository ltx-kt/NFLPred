import { NavLink, Outlet } from "react-router-dom";
import { ThemeToggle } from "./ThemeToggle";

const TABS = [
  { to: "/", label: "This week", end: true },
  { to: "/history", label: "History" },
  { to: "/model", label: "Model" },
  { to: "/about", label: "About" },
];

export function Layout() {
  return (
    <div className="min-h-full">
      <header className="border-b border-hairline bg-surface/80 backdrop-blur">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-4 py-3">
          <div className="flex items-baseline gap-2">
            <span className="text-sm font-semibold tracking-tight">NFL model</span>
            <span className="text-xs text-ink-muted">straight-up predictions</span>
          </div>
          <ThemeToggle />
        </div>
        <nav className="mx-auto flex max-w-5xl gap-1 px-3">
          {TABS.map((t) => (
            <NavLink
              key={t.to}
              to={t.to}
              end={t.end}
              className={({ isActive }) =>
                "border-b-2 px-3 py-2 text-sm transition-colors " +
                (isActive
                  ? "border-series-1 text-ink"
                  : "border-transparent text-ink-muted hover:text-ink-2")
              }
            >
              {t.label}
            </NavLink>
          ))}
        </nav>
      </header>

      <main className="mx-auto max-w-5xl px-4 py-6">
        <Outlet />
      </main>

      <footer className="mx-auto max-w-5xl px-4 py-8 text-xs text-ink-muted">
        Read-only view over the prediction log. The model refits weekly and is never
        loaded from disk. Vegas hits ~66-68% straight up.
      </footer>
    </div>
  );
}
