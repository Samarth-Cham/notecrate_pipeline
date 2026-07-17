"""
Cross-encoder reranking over hybrid candidates.

  python src/rerank.py "how do taints and tolerations work"

Pipeline: hybrid_search (top 20 candidates) -> cross-encoder scores
          every (query, chunk) pair jointly -> re-sort -> top 5.

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

RERANK_MODEL = "BAAI/bge-reranker-base"
CANDIDATES = 20   # how many hybrid results to rerank
TOP_N = 5

# Loaded once at import — first run downloads ~1GB of weights to
# ~/.cache/huggingface; subsequent runs load from disk (~5s).
_model = CrossEncoder(RERANK_MODEL)


def rerank(query: str, top_n: int = TOP_N) -> list[dict]:
    # Widen the hybrid net: temporarily lift its TOP_N to CANDIDATES
    import src.hybrid_search as hs
    original = hs.TOP_N
    hs.TOP_N = CANDIDATES
    try:
        candidates = hybrid_search(query)
    finally:
        hs.TOP_N = original

    if not candidates:
        return []

    # One (query, chunk_text) pair per candidate — scored jointly.
    pairs = [(query, c["text"]) for c in candidates]
    scores = _model.predict(pairs)          # numpy array of relevance logits

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:top_n]


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "how do taints and tolerations work"
    print(f"\nQuery: {query}\n" + "=" * 60)
    for h in rerank(query):
        ranks = f"vec#{h['vec_rank'] or '-'} txt#{h['txt_rank'] or '-'}"
        print(f"\n[{h['rerank_score']:+.3f}] ({ranks}) {h['source']}  §{h['section'] or ''}")
        print(h["text"][:250].replace("\n", " ") + "…")