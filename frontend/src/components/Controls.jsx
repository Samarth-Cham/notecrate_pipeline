const ROLES = [
  { value: "", label: "No role" },
  { value: "junior", label: "Junior" },
  { value: "senior", label: "Senior" },
];

function Toggle({ checked, onChange, children, title }) {
  return (
    <label
      className="flex cursor-pointer items-center gap-1.5 text-xs
                 text-[color:var(--color-ink-muted)]"
      title={title}
    >
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="size-3.5 accent-[color:var(--color-ink-muted)]"
      />
      {children}
    </label>
  );
}

export default function Controls({ settings, onChange, onReset, busy }) {
  const set = (patch) => onChange({ ...settings, ...patch });

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
      <label className="flex items-center gap-1.5 text-xs text-[color:var(--color-ink-muted)]">
        Role
        <select
          value={settings.role}
          onChange={(e) => set({ role: e.target.value })}
          disabled={busy}
          className="rounded border border-[color:var(--color-line)]
                     bg-[color:var(--color-surface-raised)] px-1.5 py-0.5
                     text-xs text-[color:var(--color-ink)]"
          title="In the enterprise version this comes from the authenticated identity, not a dropdown."
        >
          {ROLES.map((r) => (
            <option key={r.value} value={r.value}>{r.label}</option>
          ))}
        </select>
      </label>

      <Toggle
        checked={settings.verify}
        onChange={(v) => set({ verify: v })}
        title="Per-sentence grounding classification. Adds latency."
      >
        Grounding tags
      </Toggle>

      <Toggle
        checked={settings.route}
        onChange={(v) => set({ route: v })}
        title="Decompose multi-hop questions into sub-queries before retrieving."
      >
        Multi-hop routing
      </Toggle>

      <Toggle
        checked={settings.detect_conflicts}
        onChange={(v) => set({ detect_conflicts: v })}
        title="Known broken: 0.00 precision on a labelled set, and it degrades grounding. Off by default."
      >
        Conflict detection
        <span className="text-[color:var(--color-uncertain)]"> (broken)</span>
      </Toggle>

      <button
        type="button"
        onClick={onReset}
        className="ml-auto text-xs text-[color:var(--color-ink-muted)] underline underline-offset-2"
        title="Starts a new conversation id, clearing recalled memory."
      >
        New conversation
      </button>
    </div>
  );
}
