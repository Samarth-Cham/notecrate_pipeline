/**
 * Week 4 deliverable: the "sources disagree" panel.
 *
 * The detector behind it currently has 0.00 measured precision and is off by
 * default, so this panel renders only when a request explicitly opts in. The
 * caveat is shown inline rather than buried in a doc — a disagreement claim
 * the user cannot calibrate is worse than no claim at all.
 */
export default function Disagreement({ conflicts, sources, enabled }) {
  if (!enabled || !conflicts?.length) return null;

  const sourceName = (label) =>
    sources.find((s) => s.label === label)?.source ?? `source ${label}`;

  return (
    <section
      className="rounded-lg border border-[color:var(--color-uncertain)]/40
                 bg-[color:var(--color-uncertain)]/5 p-3"
    >
      <h3 className="flex items-center gap-2 text-sm font-semibold text-[color:var(--color-uncertain)]">
        <span className="inline-block size-2 rounded-full bg-[color:var(--color-uncertain)]" />
        Sources disagree
      </h3>

      <ul className="mt-2 space-y-2">
        {conflicts.map((c, i) => (
          <li key={i} className="text-sm">
            <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
              <cite className="not-italic font-medium">[{c.a}] {sourceName(c.a)}</cite>
              <span className="text-[color:var(--color-ink-muted)]">vs</span>
              <cite className="not-italic font-medium">[{c.b}] {sourceName(c.b)}</cite>
              <span className="text-xs text-[color:var(--color-ink-muted)]">
                p={c.score.toFixed(2)}
              </span>
            </div>
          </li>
        ))}
      </ul>

      <p className="mt-3 border-t border-[color:var(--color-uncertain)]/25 pt-2 text-xs
                    text-[color:var(--color-ink-muted)]">
        Experimental. This detector measured <strong>0.00 precision</strong> against a
        labelled set — every flagged pair was a false alarm — and enabling it costs
        faithfulness (0.61 → 0.54) and ~21s per question. Treat these as unverified.
      </p>
    </section>
  );
}
