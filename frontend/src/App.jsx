import { Fragment, useEffect, useRef, useState } from "react";
import { RefusalError, askQuestion, checkHealth } from "./api.js";
import Answer from "./components/Answer.jsx";
import Controls from "./components/Controls.jsx";
import Disagreement from "./components/Disagreement.jsx";
import Sources from "./components/Sources.jsx";

const newConversationId = () =>
  `web-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

const DEFAULT_SETTINGS = {
  role: "",
  verify: true,
  route: true,
  detect_conflicts: false,
};

function Meta({ result }) {
  const bits = [];
  if (result.route?.kind === "multi_hop") bits.push("multi-hop");
  if (result.confidence === "weak") bits.push("weak retrieval");
  if (result.grounding?.faithfulness != null) {
    bits.push(`${Math.round(result.grounding.faithfulness * 100)}% grounded`);
  }

  const stages = Object.entries(result.timings ?? {});
  const total = stages.reduce((sum, [, v]) => sum + v, 0);
  if (!bits.length && !stages.length) return null;

  return (
    <div className="flex flex-wrap items-center gap-2 text-xs
                    text-[color:var(--color-ink-muted)]">
      {bits.map((b) => (
        <span key={b} className="rounded bg-[color:var(--color-line)] px-1.5 py-0.5">
          {b}
        </span>
      ))}
      {stages.length > 0 && (
        <details className="ml-auto">
          <summary className="cursor-pointer tabular-nums">{total.toFixed(1)}s</summary>
          <dl className="mt-1 grid grid-cols-[auto_auto] gap-x-3 gap-y-0.5 text-right">
            {stages.map(([k, v]) => (
              <Fragment key={k}>
                <dt className="text-left">{k}</dt>
                <dd className="tabular-nums">{v.toFixed(2)}s</dd>
              </Fragment>
            ))}
          </dl>
        </details>
      )}
    </div>
  );
}

function Recalled({ history }) {
  if (!history?.length) return null;
  return (
    <details className="text-xs text-[color:var(--color-ink-muted)]">
      <summary className="cursor-pointer">
        Recalled {history.length} prior turn{history.length > 1 ? "s" : ""} by relevance
      </summary>
      <ul className="mt-1 space-y-0.5 pl-4">
        {history.map((h, i) => (
          <li key={i}>
            <span className="tabular-nums">
              {h.relevance != null ? h.relevance.toFixed(2) : "recent"}
            </span>
            {" — "}
            {h.question}
          </li>
        ))}
      </ul>
    </details>
  );
}

function SubQueries({ route }) {
  if (route?.kind !== "multi_hop" || !route.sub_queries?.length) return null;
  return (
    <p className="text-xs text-[color:var(--color-ink-muted)]">
      Decomposed into: {route.sub_queries.map((q) => `“${q}”`).join(", ")}
    </p>
  );
}

function Turn({ turn, settings }) {
  const [highlighted, setHighlighted] = useState(null);

  return (
    <article className="space-y-3">
      <h2 className="font-medium">{turn.question}</h2>

      {turn.state === "pending" && (
        <p className="animate-pulse text-sm text-[color:var(--color-ink-muted)]">
          Retrieving, generating, verifying…
        </p>
      )}

      {turn.state === "refused" && (
        <p className="rounded border border-[color:var(--color-line)]
                      bg-[color:var(--color-surface-raised)] p-3 text-sm
                      text-[color:var(--color-ink-muted)]">
          {turn.message}
        </p>
      )}

      {turn.state === "error" && (
        <p className="rounded border border-[color:var(--color-uncertain)]/40 p-3 text-sm
                      text-[color:var(--color-uncertain)]">
          {turn.message}
        </p>
      )}

      {turn.state === "done" && (
        <div className="space-y-4 rounded-lg border border-[color:var(--color-line)]
                        bg-[color:var(--color-surface-raised)] p-4">
          <SubQueries route={turn.result.route} />
          <Recalled history={turn.result.history} />
          <Answer result={turn.result} onHoverCitation={setHighlighted} />
          <Disagreement
            conflicts={turn.result.conflicts}
            sources={turn.result.sources}
            enabled={turn.sentWithConflicts}
          />
          <Sources
            sources={turn.result.sources}
            highlighted={highlighted}
            role={turn.result.role}
          />
          <Meta result={turn.result} />
        </div>
      )}
    </article>
  );
}

export default function App() {
  const [turns, setTurns] = useState([]);
  const [question, setQuestion] = useState("");
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [conversationId, setConversationId] = useState(newConversationId);
  const [online, setOnline] = useState(null);
  const busy = turns.some((t) => t.state === "pending");
  const bottom = useRef(null);

  useEffect(() => {
    checkHealth().then(setOnline);
  }, []);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  async function submit(e) {
    e.preventDefault();
    const q = question.trim();
    if (!q || busy) return;

    const id = Date.now();
    setQuestion("");
    setTurns((prev) => [
      ...prev,
      { id, question: q, state: "pending", sentWithConflicts: settings.detect_conflicts },
    ]);

    const finish = (patch) =>
      setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, ...patch } : t)));

    try {
      const result = await askQuestion({
        question: q,
        role: settings.role || null,
        verify: settings.verify,
        route: settings.route,
        detect_conflicts: settings.detect_conflicts,
        conversation_id: conversationId,
      });
      finish({ state: "done", result });
    } catch (err) {
      finish({
        state: err instanceof RefusalError ? "refused" : "error",
        message: err.message,
      });
    }
  }

  return (
    <div className="mx-auto flex min-h-screen max-w-3xl flex-col gap-6 px-4 py-8">
      <header className="space-y-3">
        <div className="flex items-baseline gap-3">
          <h1 className="text-xl font-semibold">NoteCrate</h1>
          <span className="text-xs text-[color:var(--color-ink-muted)]">
            multi-contextual RAG over a private corpus
          </span>
          {online === false && (
            <span className="ml-auto text-xs text-[color:var(--color-uncertain)]">
              backend unreachable
            </span>
          )}
        </div>
        <Controls
          settings={settings}
          onChange={setSettings}
          busy={busy}
          onReset={() => {
            setConversationId(newConversationId());
            setTurns([]);
          }}
        />
      </header>

      <main className="flex-1 space-y-8">
        {turns.length === 0 && (
          <div className="space-y-2 text-sm text-[color:var(--color-ink-muted)]">
            <p>Ask something the corpus covers. Try:</p>
            <ul className="list-inside list-disc space-y-1">
              <li>How does pod restart policy work?</li>
              <li>What is the difference between a Deployment and a StatefulSet?</li>
              <li>How does garbage collection work? <span className="opacity-70">— then switch role</span></li>
            </ul>
          </div>
        )}
        {turns.map((t) => (
          <Turn key={t.id} turn={t} settings={settings} />
        ))}
        <div ref={bottom} />
      </main>

      <form onSubmit={submit} className="sticky bottom-0 flex gap-2
                                         bg-[color:var(--color-surface)] py-3">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question…"
          disabled={busy}
          className="flex-1 rounded-lg border border-[color:var(--color-line)]
                     bg-[color:var(--color-surface-raised)] px-3 py-2 text-sm
                     text-[color:var(--color-ink)] outline-none
                     focus:border-[color:var(--color-ink-muted)] disabled:opacity-60"
        />
        <button
          type="submit"
          disabled={busy || !question.trim()}
          className="rounded-lg bg-[color:var(--color-ink)] px-4 py-2 text-sm font-medium
                     text-[color:var(--color-surface)] disabled:opacity-40"
        >
          {busy ? "…" : "Ask"}
        </button>
      </form>
    </div>
  );
}
