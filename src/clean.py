import re
from pathlib import Path

RAW = Path("data/raw")
CLEAN = Path("data/cleaned")
CLEAN.mkdir(parents=True, exist_ok=True)

MIN_CHARS = 200  # skip stubs

def clean_text(text: str) -> str:
    # 1. Strip YAML front matter (--- block at top)
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.DOTALL)
    # 2. Strip Hugo shortcodes: {{< ... >}} and {{% ... %}}
    text = re.sub(r"\{\{[<%].*?[>%]\}\}", "", text, flags=re.DOTALL)
    # 3. Strip HTML tags but keep inner text
    text = re.sub(r"<[^>]+>", "", text)
    # 4. Collapse 3+ blank lines into one
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

kept, skipped = 0, 0
for f in RAW.glob("*.md"):
    if f.name == "sources.md":
        continue
    cleaned = clean_text(f.read_text(encoding="utf-8", errors="ignore"))
    if len(cleaned) < MIN_CHARS:
        skipped += 1
        continue
    (CLEAN / f.name).write_text(cleaned, encoding="utf-8")
    kept += 1

print(f"Kept {kept}, skipped {skipped} stubs")