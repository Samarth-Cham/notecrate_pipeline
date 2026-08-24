"""
Cross-encoder reranking over hybrid candidates.

  python src/rerank.py "how do taints and tolerations work"

Pipeline: hybrid_search (top CANDIDATES) -> cross-encoder scores every
          (query, chunk) pair jointly -> re-sort -> top 5.

Why a cross-encoder beats the retrieval scores it replaces: bi-encoders
(nomic) embed query and chunk SEPARATELY and compare vectors — fast,
indexable, but the two texts never "see" each other. A cross-encoder
feeds the query and chunk through the model TOGETHER, attending across
them — far better relevance judgment, but O(candidates) model calls per
query, so it can only run on a small candidate set. Hence the funnel:
cheap retrieval finds 20, expensive reranker orders them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentence_transformers import CrossEncoder

from src.hybrid_search import hybrid_search
from src.permissions import UNRESTRICTED
from src.roles import adjust_score

# Chosen by measurement, not reputation. eval/tune_rerank.py sweeps model x
# candidate count against eval/questions.jsonl (26 positive questions):
#
#   model                 cand   hit@5    P@5    R@5    MRR   rerank
#   bge-reranker-base       20   1.000  0.700  0.853  0.892    7.64s
#   bge-reranker-base       10   1.000  0.777  0.865  0.904    3.64s
#   ms-marco-MiniLM-L-6     20   1.000  0.792  0.859  0.955    1.27s
#   ms-marco-MiniLM-L-6     10   1.000  0.800  0.859  0.974    0.63s   <- this
#
# 12x faster AND better on every metric, which is not the trade-off a 22M
# model against a 278M one is supposed to produce. Two things explain it:
# ms-marco-MiniLM is trained directly on passage ranking, which is exactly
# this task, and bge-reranker-base was actively HURTING — plain vector search
# scores MRR 0.973, so at 0.892 the old reranker was reordering good results
# into worse ones and charging 7.6s for it.
#
# Caveat worth keeping: 26 questions is a small set. hit@5 is 1.000 for every
# config, so the separation lives in P@5 and MRR, which a handful of questions
# can move. Re-run the sweep if the corpus changes materially.
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# 10, not 20. The funnel only needs to be wide enough that the right chunk is
# somewhere in it; scoring 20 to pick 5 was over-provisioned, and on multi-hop
# queries the cost multiplies — each sub-query gets its own rerank pass.
CANDIDATES = 10
TOP_N = 5

_model = None


def _load():
    """Loaded on first use, not at import: the first call downloads ~1GB of
    weights to ~/.cache/huggingface (later runs read from disk in ~5s). Doing
    that at import means anything that merely imports the pipeline — a unit
    test, `--help` — pays for it."""
    global _model
    if _model is None:
        _model = CrossEncoder(RERANK_MODEL)
    return _model


def rerank(query: str, top_n: int = TOP_N, qvec: list[float] = None,
           role: str = None, scopes=UNRESTRICTED) -> list[dict]:
    # Widen the hybrid net: it returns 5 by default, we want CANDIDATES to score.
    candidates = hybrid_search(query, top_n=CANDIDATES, qvec=qvec, scopes=scopes)

    if not candidates:
        return []

    # One (query, chunk_text) pair per candidate — scored jointly.
    pairs = [(query, c["text"]) for c in candidates]
    scores = _load().predict(pairs)         # numpy array of relevance logits

    for c, s in zip(candidates, scores):
        c["relevance_score"] = float(s)
        # Role conditioning happens HERE — after scoring, before the cut — so
        # a boosted chunk can actually enter the top-n. Applying it after the
        # cut would only reorder chunks that already made it, which is not the
        # feature. Both scores are kept so the effect stays auditable.
        c["rerank_score"] = adjust_score(c["relevance_score"], c.get("roles"), role)

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:top_n]


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "how do taints and tolerations work"
    print(f"\nQuery: {query}\n" + "=" * 60)
    for h in rerank(query):
        ranks = f"vec#{h['vec_rank'] or '-'} txt#{h['txt_rank'] or '-'}"
        print(f"\n[{h['rerank_score']:+.3f}] ({ranks}) {h['source']}  §{h['section'] or ''}")
        print(h["text"][:250].replace("\n", " ") + "…")