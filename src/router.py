"""
Query router / decomposer.

  python src/router.py "how do init containers differ from sidecars, and which restarts?"

Classifies an incoming question as `simple` or `multi_hop`. Multi-hop
questions get split into standalone sub-queries, each retrieved for
separately before the results are merged.

Why bother: a single embedding of "how does X differ from Y" sits
somewhere between X and Y in vector space and often retrieves neither
well. Two focused sub-queries each retrieve cleanly. This is the cheap
half of what the plan calls the Router/Decomposer stage — one small LLM
call in front of retrieval.

The router is an optimisation, never a gate: every failure path falls
back to treating the query as simple, so a bad classification costs
retrieval quality, not an error.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm import chat_json

MAX_SUB_QUERIES = 3

# Two things here are load-bearing on Llama 3.1 8B, both found by testing:
#
#   1. Examples live INLINE in the system prompt. Passed as alternating
#      user/assistant turns instead, the model ignored them — it answered
#      "simple" for a question that was verbatim one of the multi_hop examples.
#   2. "reason" is emitted BEFORE "kind". Generation is autoregressive, so
#      naming the topics first measurably improves the verdict that follows.
#
# An earlier draft also said "use simple whenever you are unsure", which
# collapsed every comparison to simple. Stating the comparison rule
# explicitly works better than an uncertainty hedge.
SYSTEM = """You split search queries for a document retrieval system.

Reply with JSON only:
{"reason": "<one short sentence>", "kind": "simple" | "multi_hop", "sub_queries": ["..."]}

Decide "kind" like this:
- "multi_hop" if answering needs information about TWO OR MORE distinct topics.
  Comparisons ("difference between A and B", "A vs B"), questions joined by
  "and", and questions that need one fact to look up another are all multi_hop.
- "simple" if one passage about ONE topic could answer it.

For "multi_hop", write one standalone retrieval query per topic (2-3 total).
Each must name its topic explicitly and use no pronouns. For "simple", use [].

Examples:
Q: How does pod restart policy work?
{"reason": "One topic: restart policy.", "kind": "simple", "sub_queries": []}

Q: What is the difference between a Deployment and a StatefulSet?
{"reason": "Comparison of two topics.", "kind": "multi_hop", "sub_queries": ["What is a Kubernetes Deployment?", "What is a Kubernetes StatefulSet?"]}

Q: Which node does the scheduler pick for a pod with a toleration?
{"reason": "Needs tolerations and scheduler behaviour.", "kind": "multi_hop", "sub_queries": ["How do taints and tolerations work?", "How does the Kubernetes scheduler assign pods to nodes?"]}"""


def route(query: str) -> dict:
    """Returns {"kind": str, "sub_queries": list[str], "queries": list[str]}.

    `queries` is what retrieval should actually run: the original question
    always comes first (it carries context the sub-queries drop), followed
    by any sub-queries.
    """
    simple = {"kind": "simple", "sub_queries": [], "queries": [query]}

    parsed = chat_json([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Q: {query}"},
    ])
    if not isinstance(parsed, dict) or parsed.get("kind") != "multi_hop":
        return simple

    subs = parsed.get("sub_queries")
    if not isinstance(subs, list):
        return simple

    # Keep only usable strings, de-duplicated, order preserved.
    seen: set[str] = set()
    clean: list[str] = []
    for s in subs:
        if not isinstance(s, str):
            continue
        s = s.strip()
        if len(s) < 8 or s.lower() in seen:
            continue
        seen.add(s.lower())
        clean.append(s)
        if len(clean) == MAX_SUB_QUERIES:
            break

    # A "multi_hop" verdict with fewer than two usable parts decomposed into
    # nothing — the original query is strictly better than one vague fragment.
    if len(clean) < 2:
        return simple

    return {"kind": "multi_hop", "sub_queries": clean, "queries": [query, *clean]}


if __name__ == "__main__":
    query = (sys.argv[1] if len(sys.argv) > 1
             else "how do init containers differ from sidecars, and which one restarts?")
    r = route(query)
    print(f"\nQuery: {query}\n" + "=" * 60)
    print(f"kind: {r['kind']}")
    for i, q in enumerate(r["sub_queries"], 1):
        print(f"  sub[{i}] {q}")
