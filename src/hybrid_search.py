"""
Hybrid search: vector + full-text, fused with Reciprocal Rank Fusion.

  python src/hybrid_search.py "how do taints and tolerations work"
"""

import os
import sys
from pathlib import Path

import psycopg
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OLLAMA = os.environ["OLLAMA_URL"]
EMBED_MODEL = os.environ["EMBED_MODEL"]
DB_URL = os.environ["DATABASE_URL"]

CANDIDATES = 20   # how deep each ranker looks
RRF_K = 60        # RRF damping constant (conventional default)
TOP_N = 5


def embed(text: str) -> list[float]:
    r = requests.post(f"{OLLAMA}/api/embeddings",
                      json={"model": EMBED_MODEL, "prompt": text})
    r.raise_for_status()
    return r.json()["embedding"]


HYBRID_SQL = """
WITH vector_hits AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> %(qvec)s::vector) AS rank,
           1 - (embedding <=> %(qvec)s::vector) AS sim
    FROM chunks
    ORDER BY embedding <=> %(qvec)s::vector
    LIMIT %(cand)s
),
text_hits AS (
    SELECT id, ROW_NUMBER() OVER (
               ORDER BY ts_rank_cd(text_search, websearch_to_tsquery('english', %(q)s)) DESC
           ) AS rank
    FROM chunks
    WHERE text_search @@ websearch_to_tsquery('english', %(q)s)
    LIMIT %(cand)s
)
SELECT c.id, c.source, c.section, c.text, c.roles,
       COALESCE(1.0 / (%(k)s + v.rank), 0) +
       COALESCE(1.0 / (%(k)s + t.rank), 0) AS rrf_score,
       v.rank AS vec_rank, t.rank AS txt_rank,
       v.sim AS vec_score,
       -- vector_hits is ordered by distance and capped at %(cand)s, so its
       -- max similarity IS the corpus-wide vector top-1. Carried on every row
       -- so the noise-floor gate works without a second embed + query.
       (SELECT MAX(sim) FROM vector_hits) AS vec_top
FROM vector_hits v
FULL OUTER JOIN text_hits t USING (id)
JOIN chunks c ON c.id = COALESCE(v.id, t.id)
ORDER BY rrf_score DESC
LIMIT %(topn)s
"""


def hybrid_search(query: str, top_n: int = None, qvec: list[float] = None) -> list[dict]:
    """Top-n chunks by RRF over vector + full-text ranks.

    `qvec` lets a caller reuse an embedding it already computed.
    """
    if qvec is None:
        qvec = embed(query)
    with psycopg.connect(DB_URL) as conn:
        rows = conn.execute(HYBRID_SQL, {
            "qvec": str(qvec), "q": query,
            "cand": CANDIDATES, "k": RRF_K,
            "topn": TOP_N if top_n is None else top_n,
        }).fetchall()
    return [
        {"id": r[0], "source": r[1], "section": r[2], "text": r[3],
         # None until src/backfill_roles.py has run; treated as "unknown
         # audience" downstream, never as a mismatch.
         "roles": r[4],
         "rrf_score": float(r[5]), "vec_rank": r[6], "txt_rank": r[7],
         # NULL when the chunk was found by keyword only, i.e. it never
         # entered the vector top-CANDIDATES.
         "vec_score": float(r[8]) if r[8] is not None else None,
         "vec_top": float(r[9]) if r[9] is not None else 0.0}
        for r in rows
    ]


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "how do taints and tolerations work"
    print(f"\nQuery: {query}\n" + "=" * 60)
    for h in hybrid_search(query):
        ranks = f"vec#{h['vec_rank'] or '-'} txt#{h['txt_rank'] or '-'}"
        print(f"\n[{h['rrf_score']:.4f}] ({ranks}) {h['source']}  §{h['section'] or ''}")
        print(h["text"][:250].replace("\n", " ") + "…")