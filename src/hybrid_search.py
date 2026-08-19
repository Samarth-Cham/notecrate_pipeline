"""
Hybrid search: vector + full-text, fused with Reciprocal Rank Fusion.

  python src/hybrid_search.py "how do taints and tolerations work"
"""

import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm import embed_query
from src.permissions import UNRESTRICTED, normalise

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DB_URL = os.environ["DATABASE_URL"]

CANDIDATES = 20   # how deep each ranker looks
RRF_K = 60        # RRF damping constant (conventional default)
TOP_N = 5


# The permission predicate is repeated in BOTH retrieval CTEs, inside each
# one's own WHERE, before its LIMIT. That placement is the security property:
# filtering afterwards would let unauthorised chunks consume candidate slots
# and reach the reranker, the conflict check and the verification pass.
#
# `%(scopes)s IS NULL` is the UNRESTRICTED path for local tools. Serving code
# always binds a real array, and an empty array matches nothing.
# The ::text[] casts are required, not cosmetic: without them Postgres cannot
# infer the parameter's type when it is NULL and raises AmbiguousParameter.
# It fails loudly rather than matching everything, which is the right
# direction for a permission predicate to break in.
_PERMITTED = ("(%(scopes)s::text[] IS NULL "
              "OR permission_scope = ANY(%(scopes)s::text[]))")

HYBRID_SQL = f"""
WITH vector_hits AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> %(qvec)s::vector) AS rank,
           1 - (embedding <=> %(qvec)s::vector) AS sim
    FROM chunks
    WHERE {_PERMITTED}
    ORDER BY embedding <=> %(qvec)s::vector
    LIMIT %(cand)s
),
text_hits AS (
    SELECT id, ROW_NUMBER() OVER (
               ORDER BY ts_rank_cd(text_search, websearch_to_tsquery('english', %(q)s)) DESC
           ) AS rank
    FROM chunks
    WHERE text_search @@ websearch_to_tsquery('english', %(q)s)
      AND {_PERMITTED}
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


def hybrid_search(query: str, top_n: int = None, qvec: list[float] = None,
                  scopes=UNRESTRICTED) -> list[dict]:
    """Top-n chunks by RRF over vector + full-text ranks.

    `qvec` lets a caller reuse an embedding it already computed.

    `scopes` is the permission filter. It defaults to UNRESTRICTED for local
    tools; every serving path goes through `answer_question`, which requires
    it explicitly. An empty list returns nothing — see src/permissions.py.
    """
    if qvec is None:
        qvec = embed_query(query)
    with psycopg.connect(DB_URL) as conn:
        rows = conn.execute(HYBRID_SQL, {
            "qvec": str(qvec), "q": query,
            "cand": CANDIDATES, "k": RRF_K,
            "topn": TOP_N if top_n is None else top_n,
            "scopes": normalise(scopes),
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