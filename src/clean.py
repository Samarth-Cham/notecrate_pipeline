import re
from pathlib import Path

RAW = Path("data/raw")
CLEAN = Path("data/cleaned")
CLEAN.mkdir(parents=True, exist_ok=True)

MIN_CHARS = 200  # skip stubs

def clean_text(text: str) -> str:
    # 1. Strip YAML front matter (--- block at top)
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.DOTALL)

    # 2. Glossary tooltips: KEEP the visible word, drop the wrapper.
    #    {{< glossary_tooltip text="node" term_id="node" >}} -> node
    text = re.sub(
        r'\{\{<\s*glossary_tooltip[^>]*?text="([^"]+)"[^>]*?>\}\}',
        r"\1",
        text,
    )

     # 2b. Tooltips with no text= attribute: fall back to the term_id as the word.
    text = re.sub(
        r'\{\{<\s*glossary_tooltip[^>]*?term_id="([^"]+)"[^>]*?>\}\}',
        r"\1",
        text,
    )

    # 3. Note/warning/tip markers: drop the markers, keep the prose inside.
    text = re.sub(r"\{\{[<%]\s*/?\s*(note|warning|caution|tip)\s*[>%]\}\}", "", text)

    # 4. All remaining shortcodes (version badges, feature-state, etc.) -> delete.
    text = re.sub(r"\{\{[<%].*?[>%]\}\}", "", text, flags=re.DOTALL)

    # 5. HTML: strip only real known tags, so placeholders like <pod-name> survive.
    text = re.sub(
        r"</?(div|span|br|p|table|tr|td|th|img|a|b|i|em|strong|code|pre|ul|ol|li|h[1-6])(\s[^>]*)?/?>",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # 6. Collapse 3+ blank lines into one
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