/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // `--plane`, `--grid` and `--axis` are used as raw CSS vars (body bg,
        // chart grid/axis strokes), not as Tailwind color classes.
        surface: "var(--surface)",
        "surface-2": "var(--surface-2)",
        ink: "var(--ink)",
        "ink-2": "var(--ink-2)",
        "ink-muted": "var(--ink-muted)",
        hairline: "var(--border)",
        "series-1": "var(--series-1)",
        "series-2": "var(--series-2)",
        "series-3": "var(--series-3)",
        good: "var(--good)",
        critical: "var(--critical)",
        flag: "var(--flag)",
      },
      fontFamily: {
        sans: ['system-ui', '-apple-system', '"Segoe UI"', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
