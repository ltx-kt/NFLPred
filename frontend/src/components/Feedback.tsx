// Shared loading / error / empty states so every page handles them the same way.

export function Spinner({ label = "Loading" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 py-12 text-ink-muted" role="status">
      <span
        className="h-4 w-4 animate-spin rounded-full border-2 border-ink-muted border-t-transparent"
        aria-hidden
      />
      <span className="text-sm">{label}...</span>
    </div>
  );
}

export function ErrorNote({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : "Something went wrong.";
  return (
    <div className="rounded-lg border border-critical/40 bg-critical/5 px-4 py-3 text-sm text-ink-2">
      <span className="font-medium text-critical">Could not load. </span>
      {message}
    </div>
  );
}

export function EmptyNote({ children }: { children: React.ReactNode }) {
  return <div className="py-12 text-sm text-ink-muted">{children}</div>;
}
