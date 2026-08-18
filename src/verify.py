"""
Verification pass: per-sentence grounding classification.

  python src/verify.py "how does pod restart policy work"

Every sentence of a generated answer is tagged:

  grounded   — a corpus chunk directly entails it
  inferred   — supported in spirit but not stated outright; a reasonable
               synthesis across chunks rather than a quotable fact
  uncertain  — nothing in the corpus supports it, or something contradicts it

The check deliberately RE-RETRIEVES per sentence rather than only scoring
against the chunks that were in the prompt. A model can restate its own
context plausibly enough to entail itself; re-retrieval asks the harder
question — is this claim in the corpus at all? Chunks the sentence cites
are added on top, so an accurate citation is never penalised by a
sentence-level retrieval miss.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hybrid_search import hybrid_search
from src.nli import score as nli_score

# Measured against eval/labels/grounding.jsonl (54 hand-labelled sentences)
# by eval/calibrate.py:
#
#   current (0.55, 0.20, 0.50)   tag accuracy 0.537, macro-F1 0.355
#   best the sweep can find      tag accuracy 0.630, macro-F1 0.420
#
# 0.55 is a reasonable cut for GROUNDED and is kept. INFERRED is not, but
# moving it will not help: entailment separates grounded from inferred with
# AUC 0.637, and their medians are 0.978 and 0.956 — the two distributions
# overlap almost completely, so no cut between them works. calibrate.py flags
# every configuration it finds, this one included, as DEGENERATE: it never
# gets `inferred` right. The three-way tag is really a two-way decision with
# a third label that never lands, and separating grounded from inferred needs
# a second signal rather than a better threshold on this one.
#
# Left unchanged pending that decision, because retuning within a broken
# parameterisation would only move which sentences are wrong. Other causes,
# written up in eval/labels/README.md: premise truncation at MAX_CHARS, the
# contradiction max() below, and attribution-shaped sentences induced by
# CONFLICT_SYSTEM — which fire only because conflict detection has 0.00
# precision (see src/conflict.py).
GROUNDED_THRESHOLD = 0.55    # entailment probability
INFERRED_THRESHOLD = 0.20    # below this, nothing meaningfully supports it
CONTRADICTED_THRESHOLD = 0.50

RETRIEVE_K = 3               # fresh chunks pulled per sentence
MIN_WORDS = 4                # shorter fragments are headings/list labels, not claims

CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

# "according to [3], the Pod stays" -> deleting just the marker leaves
# "according to, the Pod stays", which the NLI model scores as garbage.
# The attribution has to go with it.
ATTRIBUTION_RE = re.compile(
    r"\b(?:according to|as (?:stated|described|noted|mentioned|shown) in|"
    r"as per|per|based on|from|in)\s+(?:\[\d+(?:\s*,\s*\d+)*\]\s*)+,?\s*",
    re.IGNORECASE,
)

# The mirror image: "[2] states that init containers run first". The
# disagreement panel makes the model write these constantly, because
# attributing each position to its source is exactly what it was asked to do.
# Left in place, the reported claim is buried inside a claim about the corpus,
# which nothing in the corpus entails.
_REPORTING_VERB = (r"(?:states?|says?|notes?|describes?|mentions?|indicates?|"
                   r"explains?|claims?|shows?|suggests?)")

# "that" is optional: the generator writes both "[2] claims that X" and
# "[2] claims X", and the second form left a dangling verb at the front of
# the hypothesis ("claims init containers can contain utilities").
SOURCE_VERB_RE = re.compile(
    r"(?:\[\d+(?:\s*,\s*\d+)*\]\s*)+(?:\w+\s+)?" + _REPORTING_VERB
    + r"(?:\s+that)?\s+",
    re.IGNORECASE,
)

# The same shape with no citation marker at all: "another source suggests
# that X", "One source states that X". The generator writes these instead of
# "[3] suggests that X" often enough to matter — 5 of the 54 labelled
# sentences — and without a marker to anchor on, the pattern above misses
# them entirely and the claim stays wrapped in a statement about the corpus.
NAMED_SOURCE_RE = re.compile(
    r"\b(?:however,\s+)?(?:one|another|some|other|a|the)\s+sources?\s+"
    + _REPORTING_VERB + r"\s+that\s+",
    re.IGNORECASE,
)

# Don't split after these — the period is part of the token, not a sentence end.
_ABBREV = r"(?<!\be\.g)(?<!\bi\.e)(?<!\betc)(?<!\bvs)(?<!\bcf)(?<!\bDr)(?<!\bMr)(?<!\bMs)"
_SENTENCE_BREAK = re.compile(_ABBREV + r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"'(])")


def split_sentences(text: str) -> list[str]:
    """Sentence-ish segmentation, no NLP dependency.

    Newlines break first so bullets and headings stay separate units — an
    LLM answer is usually more list than prose, and a naive period-split
    glues an entire bullet list into one 'sentence'.
    """
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            out.extend(s.strip() for s in _SENTENCE_BREAK.split(line) if s.strip())
    return out


def strip_citations(sentence: str) -> str:
    """Remove citation markers and repair the punctuation they leave behind.

    Not cosmetic. Deleting "[4]" from "...of the Pod [4]." leaves "...of the
    Pod ." and the NLI model reads that dangling space as a sentence boundary:
    measured on one claim, entailment against the SAME premise dropped from
    0.744 to 0.083 purely from that one character. Re-normalise before scoring.
    """
    text = SOURCE_VERB_RE.sub("", sentence)
    text = NAMED_SOURCE_RE.sub("", text)
    text = ATTRIBUTION_RE.sub("", text)
    text = CITATION_RE.sub("", text)
    text = re.sub(r"^\s*(?:[-*+•]|\d+[.)]|#{1,6})\s+", "", text)   # list/heading markers
    text = re.sub(r"\s+([,.;:!?)])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def cited_labels(sentence: str) -> list[int]:
    """Citation markers in a sentence: '[1]', '[2][3]' and '[1, 2]' all work."""
    labels = []
    for group in CITATION_RE.findall(sentence):
        labels.extend(int(n) for n in group.split(","))
    return sorted(set(labels))


def _is_claim(sentence: str) -> bool:
    """Is this sentence something that can be true or false?

    A trailing colon means it introduces the list that follows ("A Pod can
    restart for these reasons:"). Nothing in the corpus entails a lead-in, so
    scoring one guarantees an `uncertain` tag on a sentence that asserts
    nothing — noise in the UI and a drag on the faithfulness metric.
    """
    claim = strip_citations(sentence)
    if claim.endswith(":"):
        return False
    return len(re.findall(r"[A-Za-z]{2,}", claim)) >= MIN_WORDS


def verify(answer: str, chunks: list[dict], *, retrieve_k: int = RETRIEVE_K) -> list[dict]:
    """Tag each sentence of `answer`. `chunks` are the prompt's context
    chunks, in label order — chunks[0] is what the model cites as [1]."""
    sentences = split_sentences(answer)

    # Collect every (premise, hypothesis) pair up front so the cross-encoder
    # runs once for the whole answer instead of once per sentence.
    pairs: list[tuple[str, str]] = []
    per_sentence: list[dict] = []

    for sentence in sentences:
        labels = cited_labels(sentence)
        claim = strip_citations(sentence)

        if not _is_claim(sentence):
            per_sentence.append({"text": sentence, "cited": labels,
                                 "premises": [], "span": (0, 0)})
            continue

        premises = [chunks[n - 1] for n in labels if 1 <= n <= len(chunks)]
        try:
            premises += hybrid_search(claim, top_n=retrieve_k)
        except Exception:
            # Retrieval failure degrades the check to cited-chunks-only
            # rather than failing the whole request.
            pass

        # Dedupe: a cited chunk is usually also the one re-retrieval finds.
        seen, unique = set(), []
        for p in premises:
            key = p.get("id", p["text"][:200])
            if key not in seen:
                seen.add(key)
                unique.append(p)

        start = len(pairs)
        pairs.extend((p["text"], claim) for p in unique)
        per_sentence.append({"text": sentence, "cited": labels,
                             "premises": unique, "span": (start, len(pairs))})

    scores = nli_score(pairs)

    results = []
    for item in per_sentence:
        start, end = item["span"]
        window = scores[start:end]

        if not window:
            results.append({"text": item["text"], "tag": "skipped",
                            "cited": item["cited"], "entailment": None,
                            "support": None})
            continue

        best = max(range(len(window)), key=lambda i: window[i]["entailment"])
        entail = window[best]["entailment"]
        # Contradiction is read from the SAME premise that best supports the
        # claim, not from a max over every retrieved chunk.
        #
        # The max was letting any single retrieved chunk veto a correct tag,
        # and an unrelated chunk scores contradiction ~1.00 for the reasons in
        # conflict.py. "The scheduler checks taints when making scheduling
        # decisions" is quoted verbatim from the corpus and still came out
        # `uncertain` (e=0.44, c=1.00) because two unrelated premises objected.
        # A premise that both supports the claim best AND contradicts it is
        # genuinely ambiguous; an off-topic chunk's opinion is not evidence.
        contra = window[best]["contradiction"]

        if contra >= CONTRADICTED_THRESHOLD and contra > entail:
            tag = "uncertain"
        elif entail >= GROUNDED_THRESHOLD:
            tag = "grounded"
        elif entail >= INFERRED_THRESHOLD:
            tag = "inferred"
        else:
            tag = "uncertain"

        support = item["premises"][best]
        results.append({
            "text": item["text"],
            "tag": tag,
            "cited": item["cited"],
            "entailment": round(entail, 3),
            "contradiction": round(contra, 3),
            "support": {"source": support["source"], "section": support.get("section")},
        })

    return results


def grounding_summary(sentences: list[dict]) -> dict:
    """Counts per tag plus a faithfulness ratio over classified sentences."""
    claims = [s for s in sentences if s["tag"] != "skipped"]
    counts = {tag: sum(1 for s in claims if s["tag"] == tag)
              for tag in ("grounded", "inferred", "uncertain")}
    return {
        **counts,
        "classified": len(claims),
        "faithfulness": round(counts["grounded"] / len(claims), 3) if claims else None,
    }


if __name__ == "__main__":
    from src.pipeline import answer_question

    query = sys.argv[1] if len(sys.argv) > 1 else "how does pod restart policy work"
    result = answer_question(query)

    print(f"\nQ: {query}\n" + "=" * 60)
    for s in result["sentences"]:
        mark = {"grounded": "[G]", "inferred": "[I]",
                "uncertain": "[?]", "skipped": "   "}[s["tag"]]
        conf = f" ({s['entailment']:.2f})" if s["entailment"] is not None else ""
        print(f"{mark}{conf} {s['text']}")
    print("\n" + "-" * 60)
    print(grounding_summary(result["sentences"]))
