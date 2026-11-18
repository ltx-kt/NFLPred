import { useEffect, useId, useLayoutEffect, useRef, useState, type ReactNode } from "react";

const WIDTH = 256; // px, the popover's width (w-64)
const GUTTER = 16; // px kept clear of the viewport edge

// A "?" that explains the thing next to it. Opens on hover, keyboard focus, or
// tap (touch has no hover); blur or Escape closes it. The popover is
// position: fixed and clamped to the viewport, so a tile in the left column of
// a phone grid cannot push it off-screen, and no `overflow` ancestor can clip it.
export function InfoTip({ label, children }: { label: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState({ top: 0, left: 0 });
  const button = useRef<HTMLButtonElement>(null);
  const id = useId();

  useLayoutEffect(() => {
    if (!open || !button.current) return;
    const r = button.current.getBoundingClientRect();
    // clientWidth, not innerWidth: innerWidth includes a vertical scrollbar.
    const max = document.documentElement.clientWidth - GUTTER - WIDTH;
    const centred = r.left + r.width / 2 - WIDTH / 2;
    setPos({ top: r.bottom + 6, left: Math.max(GUTTER, Math.min(centred, max)) });
  }, [open]);

  // A fixed popover would float away from its button on scroll; close instead.
  useEffect(() => {
    if (!open) return;
    const close = () => setOpen(false);
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && close();
    window.addEventListener("scroll", close, true);
    window.addEventListener("resize", close);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("scroll", close, true);
      window.removeEventListener("resize", close);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <span
      className="inline-flex"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <button
        ref={button}
        type="button"
        aria-label={`What is ${label}?`}
        aria-describedby={open ? id : undefined}
        aria-expanded={open}
        // Open, never toggle: a tap fires focus then click, so a toggle would
        // open the popover and immediately shut it. Blur or Escape closes it.
        onClick={() => setOpen(true)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-hairline text-[10px] font-semibold leading-none text-ink-muted transition-colors hover:border-ink-muted hover:text-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-series-1"
      >
        ?
      </button>
      {open && (
        <span
          id={id}
          role="tooltip"
          style={{ top: pos.top, left: pos.left, width: WIDTH }}
          className="fixed z-50 rounded-lg border border-hairline bg-surface px-3 py-2 text-left text-xs normal-case leading-relaxed tracking-normal text-ink-2 shadow-sm"
        >
          {children}
        </span>
      )}
    </span>
  );
}
