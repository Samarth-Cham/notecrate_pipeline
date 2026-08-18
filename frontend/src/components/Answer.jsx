import { useState } from "react";

// Week 4 deliverable: every sentence carries its grounding tag inline.
const TAG_STYLE = {
  grounded: {
    label: "Grounded",
    bar: "border-l-[color:var(--color-grounded)]",
    dot: "bg-[color:var(--color-grounded)]",
    text: "text-[color:var(--color-grounded)]",
    blurb: "A corpus chunk directly entails this.",
  },
  inferred: {
    label: "Inferred",
    bar: "border-l-[color:var(--color-inferred)]",
    dot: "bg-[color:var(--color-inferred)]",
    text: "text-[color:var(--color-inferred)]",
    blurb: "Supported in spirit, not stated outright.",
  },
  uncertain: {
    label: "Uncertain",
    bar: "border-l-[color:var(--color-uncertain)]",
    dot: "bg-[color:var(--color-uncertain)]",
    text: "text-[color:var(--color-uncertain)]",
    blurb: "Nothing in the corpus supports this, or something contradicts it.",
  },
};

function Citations({ labels, onHover }) {
  if (!labels?.length) return null;
  return (
    <span className="ml-1 inline-flex gap-1 align-baseline">
      {labels.map((n) => (
        <button
          key={n}
          type="button"
          onMouseEnter={() => onHover?.(n)}
          onMouseLeave={() => onHover?.(null)}
          onClick={() =>
            document
              .getElementById(`source-${n}`)
              ?.scrollIntoView({ behavior: "smooth", block: "center" })
          }
          className="rounded bg-[color:var(--color-line)] px-1.5 text-[11px] font-medium
                     text-[color:var(--color-ink-muted)] transition
                     hover:bg-[color:var(--color-ink-muted)] hover:text-[color:var(--color-surface)]"
          title={`Jump to source ${n}`}
        >
          {n}
        </button>
      ))}
    </span>
  );
}

function Sentence({ sentence, onHoverCitation }) {
  const style = TAG_STYLE[sentence.tag];

  // `skipped` covers list lead-ins and headings — sentences that assert
  // nothing, so a confidence badge on them would be noise.
  if (!style) {
    return (
      <p className="text-[color:var(--color-ink-muted)]">{sentence.text}</p>
    );
  }

  return (
    <p
      className={`border-l-2 pl-3 ${style.bar}`}
      title={`${style.label} — ${style.blurb}${
        sentence.entailment != null
          ? ` (entailment ${sentence.entailment.toFixed(2)})`
          : ""
      }${sentence.support_source ? `\nBest support: ${sentence.support_source}` : ""}`}
    >
      <span className={`mr-2 text-[10px] font-semibold uppercase tracking-wide ${style.text}`}>
        <span className={`mr-1 inline-block size-1.5 rounded-full ${style.dot}`} />
        {style.label}
      </span>
      {/* Explicit space: the badge's margin is visual only, so without this
          the copied/screen-read text reads "UNCERTAINA Pod will not...". */}
      {" "}
      {sentence.text}
      <Citations labels={sentence.cited} onHover={onHoverCitation} />
    </p>
  );
}

export default function Answer({ result, onHoverCitation }) {
  const [showRaw, setShowRaw] = useState(false);
  const tagged = result.sentences?.length > 0;

  return (
    <div className="space-y-3">
      {tagged && !showRaw ? (
        <div className="space-y-2 leading-relaxed">
          {result.sentences.map((s, i) => (
            <Sentence key={i} sentence={s} onHoverCitation={onHoverCitation} />
          ))}
        </div>
      ) : (
        <p className="whitespace-pre-wrap leading-relaxed">{result.answer}</p>
      )}

      {tagged && (
        <button
          type="button"
          onClick={() => setShowRaw((v) => !v)}
          className="text-xs text-[color:var(--color-ink-muted)] underline underline-offset-2"
        >
          {showRaw ? "Show grounding tags" : "Show plain answer"}
        </button>
      )}
    </div>
  );
}

export { TAG_STYLE };
