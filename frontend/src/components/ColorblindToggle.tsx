import { useEffect, useRef, useState } from "react";

const read = (): boolean => document.documentElement.classList.contains("cvd");

// Swaps the game-card pick bar from green/red back to the app's usual
// blue/orange pair (see the .cvd rules in index.css). Mirrors ThemeToggle's
// class-on-<html> + localStorage pattern exactly.
export function ColorblindToggle() {
  const [on, setOn] = useState<boolean>(read);
  const firstRun = useRef(true);

  useEffect(() => {
    document.documentElement.classList.toggle("cvd", on);
    if (firstRun.current) {
      firstRun.current = false;
      return;
    }
    try {
      localStorage.setItem("cvd", on ? "on" : "off");
    } catch {
      /* private mode */
    }
  }, [on]);

  return (
    <button
      type="button"
      onClick={() => setOn((v) => !v)}
      className="rounded-md border border-hairline p-1.5 text-ink-2 hover:text-ink"
      aria-label={`Turn ${on ? "off" : "on"} colorblind-friendly bar colors`}
      aria-pressed={on}
    >
      <svg viewBox="0 0 20 20" width={14} height={14} fill="currentColor" aria-hidden>
        <path d="M10 3.5C5.5 3.5 2 10 2 10s3.5 6.5 8 6.5 8-6.5 8-6.5-3.5-6.5-8-6.5Zm0 11a4.5 4.5 0 1 1 0-9 4.5 4.5 0 0 1 0 9Z" />
        <circle cx="10" cy="10" r="2.5" fill="var(--surface)" />
      </svg>
    </button>
  );
}
