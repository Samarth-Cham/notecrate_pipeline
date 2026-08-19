"""
End-to-end answer pipeline.

  route -> hybrid retrieval + rerank -> conflict detection ->
  context assembly -> Llama 3.1 -> verification pass

Everything that answers a question goes through `answer_question`, so the
CLI (ask.py), the API (api.py) and the eval harness all exercise the same
code path. Before this existed, /query still ran naive vector-only
retrieval while the reranker was only reachable from a script.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conflict import detect_conflicts
from src.llm import chat
from src.memory import as_context, recall, record_turn
from src.rerank import rerank
from src.router import route
from src.verify import verify

MAX_CONTEXT = 6      # chunks in the prompt; tokens are not the constraint, noise is
STRONG = 0.65        # above this, answer confidently

# Conflict detection is OFF by default. Not a tuning choice — the detector is
# not working. Against eval/labels/conflicts.jsonl (51 labelled pairs),
# precision is 0.00 at every threshold from 0.05 to 0.95, under both the
# min-of-both-directions rule the code uses and the max rule it replaced. At
# the shipped 0.60 it flags 14 pairs and all 14 are wrong, while all five
# genuine corpus conflicts score below 0.02.
#
# The Week 4 conclusion that requiring bidirectional agreement fixed the
# unrelated-text failure was drawn from 60 pairs. At 298 the failure is fully
# present; 60 was too few to see it.
#
# Turning it on also switches generation to CONFLICT_SYSTEM, so a false alarm
# makes the model hedge on sources that agree. Pass detect_conflicts=True to
# re-enable once the detector is rebuilt.
CONFLICT_DETECTION_ENABLED = False

# Re-fitted after adding the nomic task prefixes (src/llm.py), which shifted
# every cosine score upward. Measured on all 30 eval questions (vector top-1,
# post-routing):
#   negatives   0.562, 0.651, 0.678, 0.708
#   positives   0.704 (min), 0.712, 0.713, 0.724, ...
#
# 0.69 sits in the 0.678 -> 0.704 gap: catches 3 of 4 negatives, wrongly
# refuses none of the 26 positives.
#
# The prefixes did more than move the numbers — they SEPARATED the classes.
# The margin between the highest caught negative and the lowest positive went
# from 0.009 to 0.026, roughly 3x. Under the old unprefixed embeddings the two
# classes interleaved: "How do I set up Ollama?" scored 0.569 as a positive
# while the firewall negative scored 0.575 above it, so no threshold could
# separate them. It now scores 0.704, and is question 30 in the eval set
# precisely so that stays true.
#
# Still tuned rather than proven — 30 questions, one corpus. Re-fit whenever
# the corpus or the embedding scheme changes; changing either invalidates
# this number, and src/reembed.py prints a reminder for exactly that reason.
#
# The 4th negative (0.708) is a freshness question: on-topic for the corpus,
# but the answer postdates the export. No similarity threshold can catch that
# without refusing real questions. Generation handles it — see the refusal
# discussion in eval/run_answer_eval.py.
NOISE_FLOOR = 0.69

SYSTEM = (
    "You are an assistant answering questions from a private document corpus. "
    "Use ONLY the provided sources. If the sources do not contain the answer, "
    "say so plainly instead of guessing. Keep answers concise.\n"
    "Cite sources by putting the marker at the END of the sentence it supports, "
    'like this: "Init containers run to completion before app containers start [1]." '
    'Never write the marker mid-sentence and never write "according to [1]" — '
    "the marker alone is the attribution."
)

# Only added when conflicts are actually detected — a standing instruction
# about disagreement makes the model hedge on sources that agree fine.
CONFLICT_SYSTEM = (
    "Some of the sources below contradict each other; the contradictions are "
    "listed explicitly. Do NOT silently pick a side and do not average them "
    "into a vague statement. State what each source claims, attribute each "
    "position to its citation, and say plainly that the sources disagree."
)


def _merge(result_lists: list[list[dict]], limit: int) -> list[dict]:
    """Round-robin merge of per-sub-query results, deduped by chunk id.

    Round-robin rather than a global sort by score: for a multi-hop question,
    one sub-query usually retrieves more confidently than the other, and a
    global sort would fill the whole context with that half — which is exactly
    the failure decomposition was supposed to fix.
    """
    merged, seen = [], set()
    for rank in range(max((len(r) for r in result_lists), default=0)):
        for results in result_lists:
            if rank >= len(results):
                continue
            chunk = results[rank]
            key = chunk.get("id", chunk["text"][:200])
            if key in seen:
                continue
            seen.add(key)
            merged.append(chunk)
            if len(merged) == limit:
                return merged
    return merged


def build_prompt(query: str, chunks: list[dict], conflicts: list[dict],
                 history: list[dict] = None) -> list[dict]:
    # Label each chunk so the model can cite it. Numbered labels beat
    # filenames in the prompt: shorter, unambiguous, easy to force.
    context_blocks = []
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] {c['source']}" + (f" - {c['section']}" if c.get("section") else "")
        context_blocks.append(f"{header}\n{c['text']}")
    context = "\n\n---\n\n".join(context_blocks)

    system = SYSTEM if not conflicts else f"{SYSTEM}\n\n{CONFLICT_SYSTEM}"

    user = ""
    # History goes BEFORE the sources: it is background for interpreting the
    # question, not material to cite. Placing it after the sources invites the
    # model to cite prior answers as though they were corpus documents.
    if history:
        user += (f"Earlier in this conversation:\n\n{as_context(history)}\n\n"
                 "---\n\n")

    user += f"Sources:\n\n{context}\n\n---\n\n"
    if conflicts:
        listed = "\n".join(
            f"- [{f['a'] + 1}] and [{f['b'] + 1}] appear to contradict each other."
            for f in conflicts
        )
        user += f"Detected contradictions:\n{listed}\n\n---\n\n"
    user += f"Question: {query}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def answer_question(query: str, *, scopes, use_router: bool = True,
                    verify_answer: bool = True, role: str = None,
                    conversation_id: str = None,
                    detect_conflicts_enabled: bool = CONFLICT_DETECTION_ENABLED) -> dict:
    """Run the full pipeline. Never raises on a low-confidence query —
    returns `refused: True` so callers decide how to present it.

    `scopes` is REQUIRED and has no default. It is the permission filter, and
    a default would mean a new call site could silently retrieve the whole
    corpus. Local tools pass permissions.UNRESTRICTED explicitly; the API
    passes the scopes from the caller's token. See src/permissions.py.

    `role` conditions reranking (see src/roles.py) and is a relevance signal
    only — it is NOT access control and must never be relied on as such.
    """
    # Per-stage timings (plan section 7: latency "tracked per-stage"). Cheap to
    # collect and the only way to answer "why did that feel slow" without
    # guessing — the expensive stage is rarely the one you assume.
    timings: dict[str, float] = {}

    def _timed(name, fn):
        start = time.perf_counter()
        try:
            return fn()
        finally:
            timings[name] = round(time.perf_counter() - start, 2)

    routing = _timed("route", lambda: route(query)) if use_router else {
        "kind": "simple", "sub_queries": [], "queries": [query]}

    history = _timed("memory_recall",
                     lambda: recall(conversation_id, query)) if conversation_id else []

    per_query = _timed("retrieve", lambda: [rerank(q, role=role, scopes=scopes)
                                            for q in routing["queries"]])
    chunks = _merge(per_query, MAX_CONTEXT)

    # Refusal is gated on cosine similarity, not the reranker's logits: the
    # noise floor was calibrated in cosine space and the two are not comparable.
    vec_top = max((c["vec_top"] for c in chunks), default=0.0)

    result = {
        "question": query,
        "route": {"kind": routing["kind"], "sub_queries": routing["sub_queries"]},
        "vector_top_score": round(vec_top, 3),
        "chunks": chunks,
        "conflicts": [],
        "answer": None,
        "confidence": "weak",
        "sentences": [],
        "refused": False,
        "role": role,
        "history": history,
        "timings": timings,
    }

    if not chunks or vec_top < NOISE_FLOOR:
        result["refused"] = True
        result["reason"] = (
            f"Nothing in the corpus covers this (best match {vec_top:.3f}, "
            f"noise floor {NOISE_FLOOR})."
        )
        return result

    if detect_conflicts_enabled:
        result["conflicts"] = _timed("conflict", lambda: detect_conflicts(chunks))
    result["confidence"] = "strong" if vec_top >= STRONG else "weak"
    result["answer"] = _timed("generate", lambda: chat(
        build_prompt(query, chunks, result["conflicts"], history)))

    if verify_answer:
        result["sentences"] = _timed(
            "verify", lambda: verify(result["answer"], chunks, scopes=scopes))

    if conversation_id:
        # Recorded only on a successful answer. Storing refusals would let a
        # question the corpus cannot answer come back as "context" for the
        # next one and pollute the recall.
        _timed("memory_record",
               lambda: record_turn(conversation_id, query, result["answer"]))

    return result

    return result
