"""
Tag existing chunks with their applicable audience.

  python src/backfill_roles.py            # tag documents that have no roles yet
  python src/backfill_roles.py --force    # re-tag everything
  python src/backfill_roles.py --dry-run  # classify and print, write nothing

The plan attaches role tags at ingestion. This backfills them onto an already
indexed corpus instead, so the ingestion pipeline doesn't have to be re-run
(and 2977 chunks don't have to be re-embedded).

Classification is per DOCUMENT, not per chunk: 97 documents versus 2977 chunks
is a 30x saving, and audience is a property of how a document is written, so
the chunks of one document almost always share it. The trade-off is real
though — a "senior" deep-dive with an introductory opening section gets the
whole document's tag on that section too.

Re-running `index.py` drops and recreates the chunks table, which discards
these tags. Re-run this afterwards.
"""

import argparse
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.roles import classify_document

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
DB_URL = os.environ["DATABASE_URL"]

# Enough text to judge depth from, without paying to send a whole document.
SAMPLE_CHARS = 2500


def ensure_column(conn):
    conn.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS roles text[]")
    # Retrieval filters on this only via the rerank boost today, but the plan's
    # enterprise version does payload filtering on it.
    conn.execute("CREATE INDEX IF NOT EXISTS chunks_roles_idx ON chunks USING gin (roles)")
    conn.commit()


def documents(conn, force: bool) -> list[tuple[str, str]]:
    """(source, sample text) for each document needing classification."""
    where = "" if force else "WHERE roles IS NULL"
    rows = conn.execute(f"""
        SELECT source, string_agg(text, ' ' ORDER BY id) AS sample
        FROM chunks
        {where}
        GROUP BY source
        ORDER BY source
    """).fetchall()
    return [(r[0], (r[1] or "")[:SAMPLE_CHARS]) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-tag already-tagged documents")
    ap.add_argument("--dry-run", action="store_true", help="classify but do not write")
    args = ap.parse_args()

    with psycopg.connect(DB_URL) as conn:
        if not args.dry_run:
            ensure_column(conn)
        elif not conn.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name='chunks' AND column_name='roles'"
        ).fetchone():
            print("No roles column yet; --dry-run will classify everything.")

        docs = documents(conn, args.force or args.dry_run)
        if not docs:
            print("Nothing to tag. Use --force to re-tag.")
            return

        print(f"Classifying {len(docs)} documents...\n")
        counts: dict[str, int] = {}

        for i, (source, sample) in enumerate(docs, 1):
            tags = classify_document(source, sample)
            key = "+".join(tags)
            counts[key] = counts.get(key, 0) + 1

            if not args.dry_run:
                conn.execute("UPDATE chunks SET roles = %s WHERE source = %s",
                             (tags, source))
                conn.commit()

            print(f"  [{i:3d}/{len(docs)}] {key:15s} {source[:58]}")

        print("\nDocuments per tag:")
        for key, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {key:15s} {n}")

        if args.dry_run:
            print("\n(dry run - nothing written)")
            return

        rows = conn.execute("""
            SELECT COALESCE(array_to_string(roles, '+'), 'UNTAGGED'), count(*)
            FROM chunks GROUP BY 1 ORDER BY 2 DESC
        """).fetchall()
        print("\nChunks per tag:")
        for tag, n in rows:
            print(f"  {tag:15s} {n}")


if __name__ == "__main__":
    main()
