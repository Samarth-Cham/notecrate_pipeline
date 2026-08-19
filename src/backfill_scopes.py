"""
Assign permission scopes to an already-indexed corpus.

  python src/backfill_scopes.py            # assign scopes to untagged chunks
  python src/backfill_scopes.py --force    # reassign everything
  python src/backfill_scopes.py --dry-run

In a real deployment a chunk inherits its scope at ingestion from the ACL of
wherever it came from — the Slack channel id, the Drive folder, the repo. This
backfill stands in for that on the sandbox corpus, deriving scope from
source_type: exported chat transcripts are private, documentation is public.

Re-running `index.py` drops the chunks table, so re-run this afterwards. The
column is NOT NULL-by-default here on purpose: a chunk with no scope should be
invisible rather than universally visible, so anything this script misses
simply stops being retrievable.
"""

import argparse
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.permissions import scope_for_source_type

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
DB_URL = os.environ["DATABASE_URL"]


def ensure_column(conn) -> None:
    conn.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS permission_scope text")
    conn.execute("CREATE INDEX IF NOT EXISTS chunks_permission_scope_idx "
                 "ON chunks (permission_scope)")
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with psycopg.connect(DB_URL) as conn:
        if not args.dry_run:
            ensure_column(conn)

        where = "" if args.force else "WHERE permission_scope IS NULL"
        rows = conn.execute(
            f"SELECT source_type, count(*) FROM chunks {where} GROUP BY 1"
        ).fetchall()

        if not rows:
            print("Nothing to scope. Use --force to reassign.")
            return

        update = (
            "UPDATE chunks SET permission_scope = %s "
            "WHERE source_type IS NOT DISTINCT FROM %s"
            + ("" if args.force else " AND permission_scope IS NULL")
        )

        for source_type, n in rows:
            scope = scope_for_source_type(source_type)
            print(f"  {str(source_type):12s} -> {scope:8s} ({n} chunks)")
            if not args.dry_run:
                conn.execute(update, (scope, source_type))
        if args.dry_run:
            print("\n(dry run - nothing written)")
            return
        conn.commit()

        print("\nChunks per scope:")
        for scope, n in conn.execute(
            "SELECT COALESCE(permission_scope, 'UNSCOPED'), count(*) "
            "FROM chunks GROUP BY 1 ORDER BY 2 DESC"
        ).fetchall():
            print(f"  {scope:10s} {n}")


if __name__ == "__main__":
    main()
