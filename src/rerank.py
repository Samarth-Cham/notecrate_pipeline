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
from src.permissions import UNRESTRICTED
from src.roles import adjust_score

RERANK_MODEL = "BAAI/bge-reranker-base"
CANDIDATES = 20   # how many hybrid results to rerank
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