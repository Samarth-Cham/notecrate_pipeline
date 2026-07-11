"""
Claude conversation exports (data/raw/*.json) -> data/cleaned/chat_<title>.md

Usage:
  python src/load_chats.py            # step 1: list all conversations with indices
  python src/load_chats.py 3 7 12     # step 2: export ONLY those indices
"""

import json
import re
import sys
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/cleaned")
OUT.mkdir(parents=True, exist_ok=True)


def normalize_chatgpt(item: dict) -> dict | None:
    """ChatGPT export conversation -> Claude-like shape, or None if empty."""
    mapping = item.get("mapping", {})
    node_id = item.get("current_node")
    linear = []

    # Walk leaf -> root via parent pointers
    while node_id:
        node = mapping.get(node_id)
        if node is None:
            break
        msg = node.get("message")
        if msg:
            role = msg.get("author", {}).get("role")
            content = msg.get("content", {})
            # Only plain text turns from the actual dialogue; skip system/tool
            # turns and non-text content (code-interpreter blobs, images).
            if role in ("user", "assistant") and content.get("content_type") == "text":
                text = "\n".join(p for p in content.get("parts", [])
                                 if isinstance(p, str)).strip()
                if text:
                    linear.append({
                        "sender": "human" if role == "user" else "assistant",
                        "text": text,
                    })
        node_id = node.get("parent")

    linear.reverse()  # we walked backwards; restore chronological order
    if not linear:
        return None   # tool-only or empty conversations

    return {
        "uuid": item.get("conversation_id") or item.get("id"),
        "name": item.get("title"),
        "created_at": "",  # ChatGPT uses unix floats; skip rather than convert
        "chat_messages": linear,
    }

# --- Gather conversations from ALL json files in raw -------------------------
def load_conversations() -> list[dict]:
    convs = []
    seen_uuids = set()

    for jf in sorted(RAW.glob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  ! {jf.name}: not valid JSON, skipping")
            continue

        # Normalize both shapes to a list:
        #   conversations.json  -> already a list of conversation dicts
        #   <uuid>.json         -> a single conversation dict
        items = data if isinstance(data, list) else [data]

        for item in items:
            if not isinstance(item, dict):
                continue

            # ChatGPT export format: has a "mapping" tree instead of
            # chat_messages. Normalize it to the Claude-like shape first.
            if "mapping" in item and "chat_messages" not in item:
                item = normalize_chatgpt(item)
                if item is None:
                    continue  # empty / tool-only conversation

            # Only accept things that look like conversations
            if "chat_messages" not in item:
                continue  # skips memories.json-style objects too

            uid = item.get("uuid")
            if uid and uid in seen_uuids:
                continue  # same conversation exported in two files
            if uid:
                seen_uuids.add(uid)
            item["_from_file"] = jf.name   # provenance, handy in the listing
            convs.append(item)

    return convs


data = load_conversations()

if not data:
    print("No conversations found in data/raw/*.json")
    sys.exit(1)

# --- Step 1: no args -> list everything ---------------------------------------
if len(sys.argv) == 1:
    listing_lines = []
    for i, conv in enumerate(data):
        n_msgs = len(conv.get("chat_messages", []))
        name = conv.get("name") or "(untitled)"
        line = f"[{i:3d}] {name}  ({n_msgs} msgs, {conv.get('created_at', '?')[:10]}, from {conv['_from_file']})"
        print(line)
        listing_lines.append(line)

    listing_file = Path("data/chunks").parent / "conversation_listing.txt"  # -> data/conversation_listing.txt
    listing_file.write_text("\n".join(listing_lines), encoding="utf-8")
    print(f"\nListing saved to {listing_file}")
    print("Re-run with the indices you want, e.g.:  python src/load_chats.py 3 7 12")
    sys.exit(0)

# --- Step 2: export selected ----------------------------------------------------
def safe_filename(name: str) -> str:
    name = re.sub(r"[^\w\s-]", "", name or "").strip()
    return re.sub(r"\s+", "_", name)[:60] or "untitled"

for i in [int(a) for a in sys.argv[1:]]:
    conv = data[i]
    lines = [f"# {conv.get('name') or 'Untitled conversation'}", ""]

    for msg in conv.get("chat_messages", []):
        speaker = "Human" if msg.get("sender") == "human" else "Assistant"
        text = (msg.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"**{speaker}:** {text}")
        lines.append("")

    out_file = OUT / f"chat_{safe_filename(conv.get('name'))}.md"
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"[{i}] -> {out_file.name}  ({len(lines)} lines)")