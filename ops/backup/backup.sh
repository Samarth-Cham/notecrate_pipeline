#!/usr/bin/env bash
# Nightly vector-DB backup (plan section 3.8).
#
# Install on the staging VM:
#   crontab -e
#   15 3 * * *  /home/sam/notecrate_pipeline/ops/backup/backup.sh >> /home/sam/backup.log 2>&1
#
# The plan says "snapshots the vector DB volume nightly to object storage
# (S3/GCS bucket)". There is no cloud account in this deployment, so this
# writes to a local directory instead — which is NOT a real backup, because a
# disk failure takes the database and its backups together. It is honest
# insurance against the failure that has actually happened here twice:
# `index.py` dropping the chunks table, and `docker compose down -v` removing
# the volume. Copy BACKUP_DIR somewhere else for anything stronger.
#
# pg_dump, not a volume snapshot: a filesystem copy of a running Postgres data
# directory is not consistent, and restoring one is a gamble. pg_dump is
# transactionally consistent by construction and restores into a different
# Postgres version, which a raw volume does not.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPOSE="$REPO/docker-compose.prod.yml"
BACKUP_DIR="${BACKUP_DIR:-$HOME/notecrate-backups}"
RETAIN_DAYS="${RETAIN_DAYS:-14}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$BACKUP_DIR/notecrate_${STAMP}.dump"

mkdir -p "$BACKUP_DIR"

echo "[$(date -Is)] backup starting -> $OUT"

# -Fc (custom format): compressed, and pg_restore can filter it by table.
# Failing here must not leave a truncated file that looks like a backup, so
# write to .partial and rename only on success.
if ! docker compose -f "$COMPOSE" exec -T db \
        pg_dump -U postgres -Fc notecrate > "$OUT.partial"; then
    echo "[$(date -Is)] ERROR: pg_dump failed" >&2
    rm -f "$OUT.partial"
    exit 1
fi

# A dump of an empty database still succeeds and is ~1KB. Catching that here
# is the difference between noticing tonight and noticing during a restore.
SIZE=$(stat -c%s "$OUT.partial")
if [ "$SIZE" -lt 1000000 ]; then
    echo "[$(date -Is)] ERROR: dump is only ${SIZE} bytes; expected >1MB. Keeping as .partial for inspection." >&2
    exit 1
fi

mv "$OUT.partial" "$OUT"
echo "[$(date -Is)] wrote $(du -h "$OUT" | cut -f1)"

# Verify the archive is readable before trusting it. pg_restore -l lists the
# table of contents without restoring anything; a corrupt dump fails here.
if ! docker compose -f "$COMPOSE" exec -T db pg_restore -l /dev/stdin < "$OUT" > /dev/null 2>&1; then
    echo "[$(date -Is)] WARNING: dump written but pg_restore could not read it" >&2
fi

DELETED=$(find "$BACKUP_DIR" -name 'notecrate_*.dump' -mtime "+$RETAIN_DAYS" -print -delete | wc -l)
echo "[$(date -Is)] done. retained $(ls -1 "$BACKUP_DIR"/notecrate_*.dump 2>/dev/null | wc -l) backups, pruned $DELETED older than ${RETAIN_DAYS}d"
