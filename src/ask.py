"""
Ask a question, get a cited answer from the corpus.

  python src/ask.py "how does pod restart policy work"

Flow: embed query -> pgvector top-k -> assemble prompt with labeled
sources -> Llama 3.1 via Ollama /api/chat -> answer with citations.
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
LLM_MODEL = os.environ["LLM_MODEL"]
DB_URL = os.environ["DATABASE_URL"]

TOP_K = 5
NOISE_FLOOR = 0.57   # measured: best score for content NOT in the corpus
STRONG = 0.65        # above this, answer confidently

def embed(text: str) -> list[float]:
    r = requests.post(f"{OLLAMA}/api/embeddings",
                      json={"model": EMBED_MODEL, "prompt": text})
    r.raise_for_status()
    return r.json()["embedding"]

def retrieve(query: str) -> list[dict]:
    qvec = embed(query)
    with psycopg.connect(DB_URL) as conn:
        rows = conn.execute("""
            SELECT source, section, source_type, text,
                   1 - (embedding <=> %s::vector) AS score
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (str(qvec), str(qvec), TOP_K)).fetchall()
    return [
        {"source": r[0], "section": r[1], "source_type": r[2],
         "text": r[3], "score": r[4]}
        for r in rows
    ]

def build_prompt(query: str, chunks: list[dict]) -> list[dict]:
    # Label each chunk so the model can cite it. Numbered labels beat
    # filenames in the prompt: shorter, unambiguous, easy to force.
    context_blocks = []
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] {c['source']}" + (f" — {c['section']}" if c["section"] else "")
        context_blocks.append(f"{header}\n{c['text']}")
    context = "\n\n---\n\n".join(context_blocks)

    system = (
        "You are an assistant answering questions from a private document corpus. "
        "Use ONLY the provided sources. After every claim, cite the source it "
        "came from as [1], [2], etc. If the sources do not contain the answer, "
        "say so plainly instead of guessing. Keep answers concise."
    )
    user = f"Sources:\n\n{context}\n\n---\n\nQuestion: {query}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

def generate(messages: list[dict]) -> str:
    r = requests.post(f"{OLLAMA}/api/chat", json={
        "model": LLM_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.2},  # factual QA: low creativity
    })
    r.raise_for_status()
    return r.json()["message"]["content"]

def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "how does pod restart policy work"
    chunks = retrieve(query)

    best = chunks[0]["score"] if chunks else 0.0
    if best < NOISE_FLOOR:
        print(f"\nNothing in the corpus covers this (best match {best:.3f}, "
              f"noise floor {NOISE_FLOOR}). Not asking the model.")
        return

    if best < STRONG:
        print(f"\n[weak retrieval: best score {best:.3f} — answer may be thin]\n")

    answer = generate(build_prompt(query, chunks))

    print(f"\nQ: {query}\n" + "=" * 60)
    print(answer)
    print("\n" + "-" * 60 + "\nSources:")
    for i, c in enumerate(chunks, 1):
        sec = f" — {c['section']}" if c["section"] else ""
        print(f"  [{i}] ({c['score']:.3f}) {c['source']}{sec}")

if __name__ == "__main__":
    main()