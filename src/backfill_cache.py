import hashlib, json
from pathlib import Path
import os, psycopg
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    rows = conn.execute("SELECT text, embedding FROM chunks").fetchall()

with Path("data/embed_cache.jsonl").open("w", encoding="utf-8") as fh:
    for text, emb in rows:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        vec = json.loads(emb)          # pgvector returns '[0.1,0.2,...]' string
        fh.write(json.dumps({"h": h, "v": vec}) + "\n")

print(f"backfilled {len(rows)} embeddings")