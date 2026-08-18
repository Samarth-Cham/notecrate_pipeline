const ROLE_TAG = {
  junior: "text-[color:var(--color-grounded)]",
  senior: "text-[color:var(--color-inferred)]",
};

/**
 * Numbered source list, matching the [n] citation markers in the answer.
 *
 * When a role is active each row shows the score adjustment, so the effect of
 * role conditioning is visible rather than asserted — a boosted chunk reads
 * `0.99 (+0.08)` and a penalised one `0.89 (-0.08)`.
 */
export default function Sources({ sources, highlighted, role }) {
  if (!sources?.length) return null;

  return (
    <section>
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide
                     text-[color:var(--color-ink-muted)]">
        Sources
      </h3>
      <ol className="space-y-1.5">
        {sources.map((s) => {
          const delta = s.rerank_score - s.relevance_score;
          const adjusted = Math.abs(delta) > 1e-9;
          return (
            <li
              key={s.label}
              id={`source-${s.label}`}
              className={`flex gap-2 rounded px-2 py-1.5 text-sm transition ${
                highlighted === s.label
                  ? "bg-[color:var(--color-line)]"
                  : "bg-transparent"
              }`}
            >
              <span className="shrink-0 font-mono text-xs text-[color:var(--color-ink-muted)]">
                [{s.label}]
              </span>
              <span className="min-w-0 flex-1">
                <span className="block truncate font-medium">{s.source}</span>
                {s.section && (
                  <span className="block truncate text-xs text-[color:var(--color-ink-muted)]">
                    {s.section}
                  </span>
                )}
              </span>
              <span className="shrink-0 text-right text-xs tabular-nums
                               text-[color:var(--color-ink-muted)]">
                <span className="block">
                  {s.rerank_score.toFixed(2)}
                  {adjusted && (
                    <span className={delta > 0 ? "text-[color:var(--color-grounded)]"
                                               : "text-[color:var(--color-uncertain)]"}>
                      {" "}{delta > 0 ? "+" : ""}{delta.toFixed(2)}
                    </span>
                  )}
                </span>
                {s.roles && (
                  <span className={ROLE_TAG[s.roles[0]] ?? ""}>
                    {s.roles.join("+")}
                  </span>
                )}
              </span>
            </li>
          );
        })}
      </ol>
      {role && (
        <p className="mt-2 text-xs text-[color:var(--color-ink-muted)]">
          Scores show the role-conditioned adjustment for <strong>{role}</strong>.
          Chunks tagged <code>all</code> carry no audience signal and are not adjusted.
        </p>
      )}
    </section>
  );
}
