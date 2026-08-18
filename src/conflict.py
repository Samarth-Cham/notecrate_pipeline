"""
Pairwise NLI conflict detection over top-ranked chunks.

  python src/conflict.py "what chunk size and overlap did we choose"

For each ordered pair of retrieved chunks, classify the relationship
as entailment / contradiction / neutral. Contradictions above threshold
are surfaced so generation can present both positions instead of
silently picking one.

NLI is DIRECTIONAL — (A as premise, B as hypothesis) can score
differently from the reverse — so both orders are evaluated and a pair is
only reported when BOTH agree it is a contradiction.

Taking the stronger of the two directions (the obvious first instinct)
proved unusable: measured over 60 chunk pairs from the eval set, it flagged
7, and the clearest false positives were a Kubernetes doc paired with an
unrelated chat transcript full of PowerShell. NLI assumes its two inputs
discuss the same proposition; hand it unrelated text and it returns
confident nonsense, but usually in one direction only. Requiring agreement
looked like it fixed that — on those 60 pairs.

IT DOES NOT. eval/labels/conflicts.jsonl now exists (51 hand-labelled pairs,
drawn from the 298 the reranker actually produces across the eval set) and
eval/calibrate.py sweeps CONTRADICTION_THRESHOLD against it:

    precision is 0.00 at every threshold from 0.05 to 0.95.

Not one real conflict outranks a single negative. At the 0.60 below, 14 pairs
are flagged and all 14 are wrong — including a QuickSort trace against a
finite-state-machine truth table, scored 1.00 in BOTH directions. The
unrelated-text failure survives the min() rule at the larger sample size; 60
pairs was too few to see it.

The misses are worse than the false alarms. All five genuine corpus conflicts
score below 0.02, among them the two that retrieval really does surface
together: a chat that states India's gold import duty is ~6% next to the
chunks recording the May 2026 rise to 15%, and one chunk recommending
900-token chunks next to another recording 600 as settled. On that second
one the generator went on to assert 900/150 as the decision, which is exactly
the silent side-picking this module exists to prevent.

So the threshold is not the problem and no value of it is defensible. Until
the detector is rebuilt, the disagreement panel should be considered
non-functional rather than provisional — and note that a false positive is
not free: pipeline.py switches to CONFLICT_SYSTEM whenever this returns
anything, which measurably degrades the verification pass downstream (see
eval/labels/README.md).
"""

import sys
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nli import score as nli_score

CONTRADICTION_THRESHOLD = 0.60   # probability, not logit


def detect_conflicts(chunks: list[dict]) -> list[dict]:
    """Returns contradiction pairs above threshold, strongest first."""
    if len(chunks) < 2:
        return []

    idx_pairs = list(permutations(range(len(chunks)), 2))
    scores = nli_score([(chunks[i]["text"], chunks[j]["text"]) for i, j in idx_pairs])

    # Collapse the two directions of each pair into one score: the WEAKER of
    # them, so a pair only survives if it reads as a contradiction both ways.
    best: dict[tuple[int, int], float] = {}
    for (i, j), s in zip(idx_pairs, scores):
        key = (min(i, j), max(i, j))
        prev = best.get(key)
        best[key] = s["contradiction"] if prev is None else min(prev, s["contradiction"])

    findings = [
        {
            "a": a, "b": b,
            "score": score,
            "a_source": chunks[a]["source"],
            "b_source": chunks[b]["source"],
        }
        for (a, b), score in best.items()
        if score >= CONTRADICTION_THRESHOLD
    ]
    findings.sort(key=lambda f: f["score"], reverse=True)
    return findings


if __name__ == "__main__":
    from src.rerank import rerank

    query = sys.argv[1] if len(sys.argv) > 1 else "what chunk size and overlap did we choose"
    chunks = rerank(query)

    print(f"\nQuery: {query}\n" + "=" * 60)
    for i, c in enumerate(chunks, 1):
        print(f"[{i}] {c['source']} §{c['section'] or ''}")

    conflicts = detect_conflicts(chunks)
    if not conflicts:
        print("\nNo contradictions detected above threshold.")
    for f in conflicts:
        print(f"\n! CONFLICT (p={f['score']:.2f})  [{f['a']+1}] vs [{f['b']+1}]")
        print(f"   [{f['a']+1}] {f['a_source']}: {chunks[f['a']]['text'][:180]}...")
        print(f"   [{f['b']+1}] {f['b_source']}: {chunks[f['b']]['text'][:180]}...")
