"""
Query the index:  python src/search.py "how does pod restart policy work"
"""

import sys

import psycopg
import requests

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
DB_URL = "postgresql://postgres:notecrate@localhost:5432/notecrate"

query = sys.argv[1] if len(sys.argv) > 1 else "how do I configure pod restart policy"

# Same embedding model as indexing — mismatched models are the classic
# silent failure in RAG.
r = requests.post(f"{OLLAMA}/api/embeddings",
                  json={"model": EMBED_MODEL, "prompt": query})
qvec = r.json()["embedding"]

with psycopg.connect(DB_URL) as conn:
    with conn.cursor() as cur:
        # <=> is pgvector's cosine DISTANCE operator (0 = identical).
        # Similarity = 1 - distance, computed for familiar scoring.
        cur.execute("""
            SELECT source, section, text, 1 - (embedding <=> %s::vector) AS score
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT 5
        """, (str(qvec), str(qvec)))

        print(f"\nQuery: {query}\n" + "=" * 60)
        for source, section, text, score in cur.fetchall():
            print(f"\n[{score:.3f}] {source}  §{section or ''}")
            print(text[:300].replace("\n", " ") + "…")