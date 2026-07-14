

import json
from datetime import date, datetime, timezone
from pathlib import Path
import tiktoken
import sys

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter, 
    RecursiveCharacterTextSplitter
)

cleaned = Path("data/cleaned")
out = Path("data/chunks")
out.mkdir(parents= True, exist_ok= True)
chunk_size = int(sys.argv[1]) if len(sys.argv) > 1 else 400
overlap = int(sys.argv[2]) if len(sys.argv) > 2 else 50

# --- Tokenizer ---------------------------------------------------------------
# Size is measured in TOKENS, not characters: embedding models have token
# limits, and chars/token varies (code ~3, prose ~4). cl100k_base is a
# reasonable stand-in tokenizer.

enc = tiktoken.get_encoding("cl100k_base")

def n_tokens(text: str) -> int:
    return len(enc.encode(text))

header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "section"), ("###", "subsection")])

size_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = overlap, length_function = n_tokens, separators= ["\n\n", "\n", ".", " ", ""])

def source_type(filename: str) -> str:
    if (filename.startswith("chat_")):
        return "chat_export"
    if (filename.startswith("pdf_")):
        return "pdf"
    else:
        return "markdown" 
    
chunks_output = []
files = sorted(cleaned.glob("*.md"))

for f in files:
    text = f.read_text(encoding="utf=8")
    sections = header_splitter.split_text(text)

    for sec in sections:
        if n_tokens(sec.page_content) > chunk_size:
            pieces = size_splitter.split_text(sec.page_content)
        else:
            pieces = [sec.page_content]

        for piece in pieces:
            if n_tokens(piece) < 30:
                continue

            chunks_output.append({"text": piece,
                "metadata": {
                    "source": f.name,
                    "source_type": source_type(f.name),
                    "section": sec.metadata.get("section", ""),
                    "subsection": sec.metadata.get("subsection", ""),
                    "chunk_strategy": "structural",   # for the level-4 A/B later

                    # member-model fields (stubbed for the test corpus)
                    "source_id": f"file:{f.stem}",                     # stable ID; later: Slack channel ID ("C0123ABC") / Notion page ID
                    "origin": f"local/{f.parent.name}/{f.name}",       # human-readable location; later: "#eng-backend" or "Notion / Onboarding"
                    "author": "samarth",                               # later: Slack user ID ("U0456DEF") / Notion last_edited_by
                    "created_at": datetime.fromtimestamp(
                        f.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),                                     # content's own timestamp; later: message ts / page last_edited_time
                    "permission_scope": "test",                        # later: inherited from the invitation (e.g. the channel ID itself)

                    "owner": "notecrate-test-corpus",
                    "ingested": date.today().isoformat(),
                    "n_tokens": n_tokens(piece),
},
            })

output_file = out/"chunks.jsonl"
with output_file.open("w", encoding="utf-8") as fh:
    for c in chunks_output:
        fh.write(json.dumps(c, ensure_ascii= False) + "\n")

sizes = [c["metadata"]["n_tokens"] for c in chunks_output]
sizes.sort()
print(f"{len(chunks_output)} chunks from {len(files)} docs -> {output_file}")
print(f"tokens/chunk: min={sizes[0]}  median={sizes[len(sizes)//2]}  "
      f"max={sizes[-1]}  (target={chunk_size}, overlap={overlap})")
