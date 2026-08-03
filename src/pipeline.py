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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conflict import detect_conflicts
from src.llm import chat
from src.rerank import rerank
from src.router import route
from src.verify import verify

MAX_CONTEXT = 6      # chunks in the prompt; tokens are not the constraint, noise is
NOISE_FLOOR = 0.57   # measured: best vector score for content NOT in the corpus
STRONG = 0.65        # above this, answer confidently

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


def build_prompt(query: str, chunks: list[dict], conflicts: list[dict]) -> list[dict]:
    # Label each chunk so the model can cite it. Numbered labels beat
    # filenames in the prompt: shorter, unambiguous, easy to force.
    context_blocks = []
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] {c['source']}" + (f" - {c['section']}" if c.get("section") else "")
        context_blocks.append(f"{header}\n{c['text']}")
    context = "\n\n---\n\n".join(context_blocks)

    system = SYSTEM if not conflicts else f"{SYSTEM}\n\n{CONFLICT_SYSTEM}"

    user = f"Sources:\n\n{context}\n\n---\n\n"
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


def answer_question(query: str, *, use_router: bool = True,
                    verify_answer: bool = True) -> dict:
    """Run the full pipeline. Never raises on a low-confidence query —
    returns `refused: True` so callers decide how to present it."""
    routing = route(query) if use_router else {"kind": "simple", "sub_queries": [],
                                               "queries": [query]}

    per_query = [rerank(q) for q in routing["queries"]]
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
    }

    if not chunks or vec_top < NOISE_FLOOR:
        result["refused"] = True
        result["reason"] = (
            f"Nothing in the corpus covers this (best match {vec_top:.3f}, "
            f"noise floor {NOISE_FLOOR})."
        )
        return result

    result["conflicts"] = detect_conflicts(chunks)
    result["confidence"] = "strong" if vec_top >= STRONG else "weak"
    result["answer"] = chat(build_prompt(query, chunks, result["conflicts"]))

    if verify_answer:
        result["sentences"] = verify(result["answer"], chunks)

    return result
