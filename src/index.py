"""
chunks.jsonl -> embeddings (Ollama, cached) -> Postgres/pgvector

Run AFTER chunk.py. Re-running drops and recreates the table
(clean slate each run, no stale chunks).

Embedding cache: data/embed_cache.jsonl maps sha256(chunk text) -> vector.
Unchanged chunks are free on re-runs; only new/modified text hits Ollama.
"""

import hashlib
import json
import os
from pathlib import Path

import psycopg
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OLLAMA = os.environ["OLLAMA_URL"]
EMBED_MODEL = os.environ["EMBED_MODEL"]
DB_URL = os.environ["DATABASE_URL"]

CHUNKS = Path("data/chunks/chunks.jsonl")
CACHE_FILE = Path("data/embed_cache.jsonl")


# --- embedding + cache --------------------------------------------------------

def embed(text: str) -> list[float]:
    r = requests.post(f"{OLLAMA}/api/embeddings",
                      json={"model": EMBED_MODEL, "prompt": text})
    r.raise_for_status()
    return r.json()["embedding"]


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_cache() -> dict[str, list[float]]:
    """text-hash -> embedding, accumulated across previous runs."""
    cache = {}
    if CACHE_FILE.exists():
        with CACHE_FILE.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                cache[rec["h"]] = rec["v"]
    return cache


# --- load chunks -----------------------------------------------------------------

chunks = [json.loads(line) for line in CHUNKS.open(encoding="utf-8")]
print(f"{len(chunks)} chunks to index")

cache = load_cache()
print(f"embedding cache: {len(cache)} entries loaded")

# --- index -----------------------------------------------------------------------

with psycopg.connect(DB_URL) as conn:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

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

        BATCH = 64
        batch = []
        hits, misses = 0, 0

        with CACHE_FILE.open("a", encoding="utf-8") as cache_fh:
            for i, chunk in enumerate(chunks):
                h = text_hash(chunk["text"])
                if h in cache:
                    vec = cache[h]
                    hits += 1
                else:
                    vec = embed(chunk["text"])
                    cache[h] = vec
                    cache_fh.write(json.dumps({"h": h, "v": vec}) + "\n")
                    misses += 1

                m = chunk["metadata"]
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
                    print(f"  indexed {i + 1}/{len(chunks)}  "
                          f"(cache hits {hits}, embedded {misses})")

            if batch:
                cur.executemany(
                    "INSERT INTO chunks VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    batch,
                )
                conn.commit()

        # HNSW index built AFTER bulk insert — much faster than
        # maintaining it during inserts.
        cur.execute("""
            CREATE INDEX ON chunks
            USING hnsw (embedding vector_cosine_ops)
        """)
        conn.commit()

        cur.execute("SELECT count(*) FROM chunks")
        total = cur.fetchone()[0]

print(f"Done. chunks table has {total} rows  "
      f"(cache hits {hits}, embedded fresh {misses})")