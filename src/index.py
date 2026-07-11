"""
chunks.jsonl -> embeddings (Ollama) -> Postgres/pgvector

Run AFTER chunk.py. Re-running drops and recreates the table
(clean slate each run, no stale chunks).
"""

import json
from pathlib import Path

import psycopg
import requests

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
DB_URL = "postgresql://postgres:notecrate@localhost:5432/notecrate"

CHUNKS = Path("data/chunks/chunks.jsonl")

def embed(text: str) -> list[float]:
    r = requests.post(f"{OLLAMA}/api/embeddings",
                      json={"model": EMBED_MODEL, "prompt": text})
    r.raise_for_status()
    return r.json()["embedding"]

chunks = [json.loads(line) for line in CHUNKS.open(encoding="utf-8")]
print(f"{len(chunks)} chunks to index")

with psycopg.connect(DB_URL) as conn:
    with conn.cursor() as cur:
        # Enable the extension (no-op if already enabled)
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

        # Clean slate: metadata as JSONB keeps the schema stable even if
        # chunk.py's metadata fields evolve; promoted columns are the ones
        # you'll filter on.
        cur.execute("DROP TABLE IF EXISTS chunks")
        cur.execute("""
            CREATE TABLE chunks (
                id          integer PRIMARY KEY,
                text        text NOT NULL,
                source      text,
                source_type text,
                section     text,
                metadata    jsonb,
                embedding   vector(768)
            )
        """)

        # Embed + insert in batches
        BATCH = 64
        batch = []
        for i, chunk in enumerate(chunks):
            m = chunk["metadata"]
            vec = embed(chunk["text"])
            batch.append((
                i,
                chunk["text"],
                m.get("source"),
                m.get("source_type"),
                m.get("section"),
                json.dumps(m),
                vec,
            ))
            if len(batch) >= BATCH:
                cur.executemany(
                    "INSERT INTO chunks VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    batch,
                )
                conn.commit()
                batch = []
                print(f"  indexed {i + 1}/{len(chunks)}")

        if batch:
            cur.executemany(
                "INSERT INTO chunks VALUES (%s, %s, %s, %s, %s, %s, %s)",
                batch,
            )
            conn.commit()

        # HNSW index for fast cosine search. Built AFTER inserts —
        # bulk-load-then-index is much faster than maintaining the
        # index during inserts.
        cur.execute("""
            CREATE INDEX ON chunks
            USING hnsw (embedding vector_cosine_ops)
        """)
        conn.commit()

        cur.execute("SELECT count(*) FROM chunks")
        print(f"Done. chunks table has {cur.fetchone()[0]} rows")