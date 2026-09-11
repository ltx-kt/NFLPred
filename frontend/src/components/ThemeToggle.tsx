import { useEffect, useRef, useState } from "react";

type Theme = "light" | "dark";

const read = (): Theme =>
  document.documentElement.classList.contains("dark") ? "dark" : "light";

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(read);
  const firstRun = useRef(true);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    // Skip the write on mount: reading the OS-inherited class here would silently
    // pin "follow system" into an explicit choice before the user ever clicked.
    if (firstRun.current) {
      firstRun.current = false;
      return;
    }
    try {
      localStorage.setItem("theme", theme);
    } catch {
      /* private mode */
    }
  }, [theme]);

  return (
    <button
      type="button"
      onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
      className="rounded-md border border-hairline p-1.5 text-ink-2 hover:text-ink"
      aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
      aria-pressed={theme === "dark"}
    >
      {theme === "dark" ? (
        <svg viewBox="0 0 20 20" width={14} height={14} fill="currentColor" aria-hidden>
          <path d="M17.293 13.293A8 8 0 0 1 6.707 2.707a8.001 8.001 0 1 0 10.586 10.586Z" />
        </svg>
      ) : (
        <svg viewBox="0 0 20 20" width={14} height={14} fill="currentColor" aria-hidden>
          <path d="M10 4a1 1 0 0 1 1-1V1a1 1 0 1 0-2 0v2a1 1 0 0 1 1 1Zm0 12a1 1 0 0 1 1 1v2a1 1 0 1 1-2 0v-2a1 1 0 0 1 1-1Zm7-6a1 1 0 0 1-1 1h-2a1 1 0 1 1 0-2h2a1 1 0 0 1 1 1ZM6 10a1 1 0 0 1-1 1H3a1 1 0 1 1 0-2h2a1 1 0 0 1 1 1Zm8.657-5.657a1 1 0 0 1 0 1.414l-1.414 1.415a1 1 0 1 1-1.415-1.415l1.415-1.414a1 1 0 0 1 1.414 0ZM7.172 14.243a1 1 0 0 1 0 1.414l-1.415 1.415a1 1 0 1 1-1.414-1.415l1.414-1.414a1 1 0 0 1 1.415 0Zm8.9 2.829a1 1 0 0 1-1.415 0l-1.414-1.415a1 1 0 1 1 1.414-1.414l1.415 1.414a1 1 0 0 1 0 1.415ZM5.757 5.757a1 1 0 0 1-1.414 0L2.929 4.343a1 1 0 1 1 1.414-1.414l1.414 1.414a1 1 0 0 1 0 1.414ZM10 6a4 4 0 1 0 0 8 4 4 0 0 0 0-8Z" />
        </svg>
      )}
    </button>
  );
}
