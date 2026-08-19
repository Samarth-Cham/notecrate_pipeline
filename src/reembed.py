"""
Re-embed the indexed corpus in place.

  python src/reembed.py            # chunks + conversation turns
  python src/reembed.py --dry-run

Use this when the EMBEDDING SCHEME changes but the text does not — adding the
nomic task prefixes, or switching embedding model. It UPDATEs vectors on the
existing rows rather than rebuilding the table, so everything else survives:
`roles` from backfill_roles.py, `permission_scope` from backfill_scopes.py,
and the chunk ids that conversation history and saved eval results refer to.

Running `index.py` instead would DROP the table and silently discard all of
that.

Conversation turns are re-embedded too. They are stored with the DOCUMENT
prefix and recalled with the QUERY prefix; leaving them on the old scheme
while queries move to the new one would quietly degrade memory recall.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import psycopg
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm import DOCUMENT_PREFIX, EMBED_MODEL, embed_document

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
DB_URL = os.environ["DATABASE_URL"]

BATCH = 100     # rows committed per transaction


def _reembed(conn, table: str, id_col: str, text_expr: str, dry_run: bool) -> int:
    rows = conn.execute(f"SELECT {id_col}, {text_expr} FROM {table} ORDER BY {id_col}").fetchall()
    if not rows:
        print(f"  {table}: nothing to do")
        return 0
    if dry_run:
        print(f"  {table}: would re-embed {len(rows)} rows")
        return 0

    started = time.perf_counter()
    for n, (row_id, text) in enumerate(rows, 1):
        conn.execute(f"UPDATE {table} SET embedding = %s WHERE {id_col} = %s",
                     (str(embed_document(text)), row_id))
        if n % BATCH == 0:
            conn.commit()
            rate = n / (time.perf_counter() - started)
            eta = (len(rows) - n) / rate
            print(f"  {table}: {n}/{len(rows)}  ({rate:.0f}/s, ~{eta:.0f}s left)")
    conn.commit()
    print(f"  {table}: {len(rows)} rows in {time.perf_counter() - started:.0f}s")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"Model  : {EMBED_MODEL}")
    print(f"Prefix : {DOCUMENT_PREFIX!r}\n")

    with psycopg.connect(DB_URL) as conn:
        _reembed(conn, "chunks", "id", "text", args.dry_run)

        turns_exist = conn.execute(
            "SELECT to_regclass('public.turns') IS NOT NULL").fetchone()[0]
        if turns_exist:
            _reembed(conn, "turns", "id",
                     "question || E'\\n\\n' || answer", args.dry_run)

    if args.dry_run:
        print("\n(dry run - nothing written)")
        return

    print("\nDone. The noise floor is calibrated in cosine space, so it no "
          "longer matches this distribution — re-fit it before trusting "
          "refusals (see NOISE_FLOOR in src/pipeline.py).")


if __name__ == "__main__":
    main()
