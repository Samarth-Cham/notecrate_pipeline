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

export default function Controls({ settings, onChange, onReset }) {
  const set = (patch) => onChange({ ...settings, ...patch });

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
      {/* No role selector here any more — it is a property of the signed-in
          identity, read from the token server-side (plan §2.3). */}
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
