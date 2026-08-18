"""
Role-aware retrieval: taxonomy, tagging, and score conditioning.

Per the plan (section 2.3), the same corpus serves different audiences by
RE-SCORING retrieval results, not by maintaining separate knowledge bases and
not by filtering. A senior asking a beginner question must still get the
beginner document — it just has to compete slightly harder.

Roles are an applicable-audience list on each chunk:

  all      conceptual material useful to anyone
  junior   setup guides, tutorials, walkthroughs, basic definitions
  senior   internals, edge cases, performance, security, design trade-offs

A chunk matches a user when its list contains "all" or the user's own role.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm import chat_json

ROLES = ("junior", "senior")
ALL = "all"
VALID_TAGS = frozenset((*ROLES, ALL))

# Calibrated against the observed cross-encoder score spread. bge-reranker
# scores are sigmoid-squashed into (0,1), and the median gap from rank 1 to
# rank 5 is 0.281 — roughly 0.07 per position. So 0.08 is worth about one
# position inside the top-5 band: enough to reorder near-ties by audience,
# not enough to promote an irrelevant chunk over a relevant one. That is the
# whole design constraint; raising this turns a boost into a filter.
ROLE_BOOST = 0.08
ROLE_PENALTY = 0.08

# "all" is deliberately NOT boosted. It means "no audience signal", which is
# the same information content as an untagged chunk — so it gets no
# adjustment, and the boost/penalty measure movement relative to it.
# An earlier version boosted "all" equally with an exact match. Since "all" is
# half the corpus, that applied a near-uniform shift to every candidate, and a
# uniform shift cannot reorder anything: role had no effect on the top-5 for
# any query where the audience-specific material wasn't already winning.

CLASSIFY_SYSTEM = """You label a document with the audience it is written for.

Reply with JSON only:
{"reason": "<one short sentence>", "roles": ["all"] | ["junior"] | ["senior"] | ["junior", "senior"]}

- "junior": setup instructions, tutorials, getting-started walkthroughs, basic
  definitions, debugging a common first-time error.
- "senior": internal mechanics, edge cases, performance characteristics,
  security considerations, architectural trade-offs, tuning.
- "all": conceptual explanations useful at any level, or documents that clearly
  serve both audiences.

Judge the DEPTH of treatment, not the topic. A beginner walkthrough of a hard
topic is "junior"; a discussion of failure modes in an easy topic is "senior".

Examples:
Document: "How to install the CLI, then run `init` to create your first project..."
{"reason": "Step-by-step setup walkthrough.", "roles": ["junior"]}

Document: "The scheduler's preemption path recomputes victim sets per node, which is O(n*m) and degrades above ~5k pods..."
{"reason": "Internal mechanics and performance limits.", "roles": ["senior"]}

Document: "A Deployment manages a replicated application. It creates a ReplicaSet, which creates Pods..."
{"reason": "Conceptual overview useful at any level.", "roles": ["all"]}"""


def classify_document(source: str, sample: str) -> list[str]:
    """Label one document. Falls back to ["all"] — the neutral tag — so a
    classification failure never silently biases retrieval."""
    parsed = chat_json([
        {"role": "system", "content": CLASSIFY_SYSTEM},
        {"role": "user", "content": f"Document: {source}\n\n{sample[:2500]}"},
    ])
    if not isinstance(parsed, dict):
        return [ALL]

    tags = [t for t in parsed.get("roles", [])
            if isinstance(t, str) and t.lower() in VALID_TAGS]
    if not tags:
        return [ALL]

    # Tagged for every role is the same statement as "all", and storing it
    # one way keeps the match test simple.
    tags = sorted(set(t.lower() for t in tags))
    return [ALL] if set(ROLES).issubset(tags) else tags


def matches(chunk_roles: list[str] | None, user_role: str | None) -> bool | None:
    """True on an exact audience match, False on a mismatch, None when there
    is no signal either way — no role requested, chunk untagged, or the chunk
    is tagged "all".

    None is distinct from False on purpose: an untagged chunk should be left
    alone, not penalised. Otherwise the feature silently degrades retrieval
    for any corpus where the backfill hasn't run.
    """
    if not user_role or not chunk_roles or ALL in chunk_roles:
        return None
    return user_role in chunk_roles


def adjust_score(score: float, chunk_roles: list[str] | None,
                 user_role: str | None) -> float:
    """Apply the role-conditioned boost/penalty to a rerank score."""
    m = matches(chunk_roles, user_role)
    if m is None:
        return score
    return score + (ROLE_BOOST if m else -ROLE_PENALTY)
