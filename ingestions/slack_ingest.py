"""
Slack member-model backfill: one channel -> chunks.jsonl-compatible output.

  python src/slack_ingest.py            # list channels the bot can see
  python src/slack_ingest.py C0123ABC   # backfill that channel

Member model: the bot only sees channels it was invited to.
This script ingests exactly one invited channel per run.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])

OUT = ROOT / "data" / "slack"
OUT.mkdir(parents=True, exist_ok=True)


# --- helpers -----------------------------------------------------------------

def list_bot_channels() -> list[dict]:
    """Channels the bot is a MEMBER of — the member-model boundary."""
    channels = []
    cursor = None
    while True:
        resp = client.conversations_list(types="public_channel",
                                         limit=200, cursor=cursor)
        channels.extend(c for c in resp["channels"] if c.get("is_member"))
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    return channels


_user_cache: dict[str, str] = {}

def resolve_user(user_id: str) -> str:
    """User ID -> display name, cached (users:read scope)."""
    if user_id not in _user_cache:
        try:
            info = client.users_info(user=user_id)
            profile = info["user"]["profile"]
            _user_cache[user_id] = (profile.get("display_name")
                                    or profile.get("real_name") or user_id)
        except SlackApiError:
            _user_cache[user_id] = user_id
    return _user_cache[user_id]


def fetch_history(channel_id: str) -> list[dict]:
    """Full message history via cursor pagination, rate-limit-politely."""
    messages = []
    cursor = None
    while True:
        try:
            resp = client.conversations_history(channel=channel_id,
                                                limit=200, cursor=cursor)
        except SlackApiError as e:
            if e.response.status_code == 429:          # rate limited
                wait = int(e.response.headers.get("Retry-After", 30))
                print(f"  rate limited — sleeping {wait}s")
                time.sleep(wait)
                continue
            raise
        messages.extend(resp["messages"])
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(1.2)   # stay under Tier 3 limits without drama
    return messages


# --- main --------------------------------------------------------------------

if len(sys.argv) == 1:
    print("Channels the bot is a member of:\n")
    for c in list_bot_channels():
        print(f"  {c['id']}  #{c['name']}")
    print("\nRe-run with a channel ID to backfill it.")
    sys.exit(0)

channel_id = sys.argv[1]
info = client.conversations_info(channel=channel_id)["channel"]
channel_name = info["name"]

if not info.get("is_member"):
    sys.exit(f"Bot is not a member of #{channel_name} — invite it first. "
             "(Member model: no invitation, no ingestion.)")

print(f"Backfilling #{channel_name} ({channel_id})...")
raw = fetch_history(channel_id)
print(f"  {len(raw)} raw messages")

# Normalize: skip non-user noise (joins, bots), newest-last ordering
records = []
for m in reversed(raw):                       # Slack returns newest first
    if m.get("type") != "message" or m.get("subtype"):
        continue                              # joins/leaves/bot_message etc.
    text = (m.get("text") or "").strip()
    if not text:
        continue
    ts = float(m["ts"])
    records.append({
        "text": f"{resolve_user(m.get('user', '?'))}: {text}",
        "metadata": {
            "source": f"slack_{channel_name}",
            "source_type": "slack",
            "source_id": channel_id,           # the permission boundary key
            "origin": f"#{channel_name}",
            "author": resolve_user(m.get("user", "?")),
            "created_at": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
            "permission_scope": channel_id,    # invitation-derived, per plan
            "slack_ts": m["ts"],               # native ID — dedupe/updates later
        },
    })

out_file = OUT / f"{channel_name}.jsonl"
with out_file.open("w", encoding="utf-8") as fh:
    for r in records:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"  {len(records)} messages -> {out_file}")