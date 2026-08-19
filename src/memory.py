"""
Conversation memory: store prior turns, retrieve the RELEVANT ones.

  python src/memory.py <conversation_id>     # show a conversation

The naive approach is to append the whole transcript to every prompt. That
fails in two directions at once: it burns context on turns that have nothing
to do with the current question, and it still drops the one exchange from
twenty turns ago that actually mattered once the window fills.

So turns are embedded and stored, and each new question retrieves the k most
semantically relevant prior turns instead of the k most recent. Recency is
kept only as a tie-breaker.

The last turn is always included regardless of score, because follow-ups like
"what about the other one?" carry almost no retrievable signal of their own —
they are only interpretable against the turn immediately before them.
"""

import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm import embed_document, embed_query

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
DB_URL = os.environ["DATABASE_URL"]

TOP_TURNS = 3          # relevant prior turns pulled into context
RECENT_ALWAYS = 1      # most recent turns included unconditionally

# Below this, a prior turn is padding. Measured on a mixed-topic conversation:
# an unrelated turn (Kubernetes question, quick-sort history) scored 0.35,
# while a genuine follow-up matched its target at 0.57.
#
# The separation is NOT clean. Two unrelated Kubernetes turns still scored
# 0.49, because turns are embedded as question+answer and any two answers from
# the same domain share vocabulary. So this reliably drops cross-domain noise
# and does not reliably drop same-domain noise. RECENT_ALWAYS is what actually
# guarantees a follow-up can resolve its antecedent; this is just hygiene.
MIN_RELEVANCE = 0.45

SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id              bigserial PRIMARY KEY,
    conversation_id text NOT NULL,
    question        text NOT NULL,
    answer          text NOT NULL,
    embedding       vector(768) NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS turns_conversation_idx ON turns (conversation_id, id);
"""


def ensure_schema(conn):
    conn.execute(SCHEMA)
    conn.commit()


def record_turn(conversation_id: str, question: str, answer: str) -> None:
    """Store a completed exchange.

    The embedding covers question AND answer together: a follow-up often
    matches vocabulary the user never typed, only the assistant did.
    """
    vec = embed_document(f"{question}\n\n{answer}")
    with psycopg.connect(DB_URL) as conn:
        ensure_schema(conn)
        conn.execute(
            "INSERT INTO turns (conversation_id, question, answer, embedding) "
            "VALUES (%s, %s, %s, %s)",
            (conversation_id, question, answer, str(vec)),
        )
        conn.commit()


def recall(conversation_id: str, query: str, *, top_k: int = TOP_TURNS,
           qvec: list[float] = None) -> list[dict]:
    """Prior turns worth showing the model, oldest first.

    Returned in chronological order even though they were selected by
    relevance — a transcript that jumps around in time reads as though the
    conversation happened out of order, and the model follows that cue.
    """
    if not conversation_id:
        return []
    if qvec is None:
        qvec = embed_query(query)

    with psycopg.connect(DB_URL) as conn:
        ensure_schema(conn)
        rows = conn.execute("""
            SELECT id, question, answer, relevance FROM (
                SELECT id, question, answer,
                       1 - (embedding <=> %(qvec)s::vector) AS relevance
                FROM turns
                WHERE conversation_id = %(cid)s
            ) scored
            WHERE relevance >= %(floor)s
            ORDER BY relevance DESC, id DESC
            LIMIT %(k)s
        """, {"qvec": str(qvec), "cid": conversation_id, "k": top_k,
              "floor": MIN_RELEVANCE}).fetchall()

        # The most recent turn is mandatory context for follow-ups, so if
        # relevance ranking left it out, pull it in explicitly.
        selected = {r[0] for r in rows}
        recent = conn.execute("""
            SELECT id, question, answer, NULL::float
            FROM turns WHERE conversation_id = %s ORDER BY id DESC LIMIT %s
        """, (conversation_id, RECENT_ALWAYS)).fetchall()

    combined = list(rows) + [r for r in recent if r[0] not in selected]
    combined.sort(key=lambda r: r[0])   # chronological for the prompt
    return [{"id": r[0], "question": r[1], "answer": r[2],
             "relevance": round(r[3], 3) if r[3] is not None else None}
            for r in combined]


def as_context(turns: list[dict]) -> str:
    """Render recalled turns for the prompt."""
    return "\n\n".join(
        f"Earlier question: {t['question']}\nEarlier answer: {t['answer']}"
        for t in turns
    )


if __name__ == "__main__":
    cid = sys.argv[1] if len(sys.argv) > 1 else "demo"
    with psycopg.connect(DB_URL) as conn:
        ensure_schema(conn)
        rows = conn.execute(
            "SELECT id, question, created_at FROM turns "
            "WHERE conversation_id = %s ORDER BY id", (cid,)).fetchall()
    print(f"\nConversation {cid!r}: {len(rows)} turns")
    for r in rows:
        print(f"  [{r[0]}] {r[2]:%Y-%m-%d %H:%M}  {r[1][:64]}")
