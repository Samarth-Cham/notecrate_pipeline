"""
Build the candidate pools that the hand-labelled validation set is drawn from.

  python eval/harvest_labels.py conflicts          # chunk pairs + NLI scores
  python eval/harvest_labels.py answers            # freeze answers, then sentences
  python eval/harvest_labels.py answers --limit 12

This script does NOT assign labels. It collects the material a human reads —
chunk pairs that the retriever actually puts in front of the detector, and
sentences the generator actually wrote — and writes it to `_pool_*.jsonl`
files. Labels are then entered by hand into conflicts.jsonl / grounding.jsonl,
which are the artifacts that matter and the ones under version control.

Why a frozen answers.jsonl instead of regenerating:

  Generation runs at temperature 0.2. Non-zero means a re-run produces
  different sentences, and a grounding label attached to a sentence that no
  longer exists is worthless. So answers are generated ONCE, written to
  eval/labels/answers.jsonl, and every later step — labelling, calibration,
  re-calibration after an NLI model swap — reads that file. Regenerating is a
  deliberate act (`answers --regenerate`) that invalidates the labels, and the
  script says so before it does it.

The pools carry the scores the current code would produce, purely so a
labeller can see what the detector thinks. Read the text and decide first;
the score column is there to make disagreements easy to find afterwards.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nli import score as nli_score
from src.permissions import UNRESTRICTED
from src.pipeline import answer_question
from src.rerank import rerank
from src.verify import RETRIEVE_K, _is_claim, cited_labels, split_sentences, strip_citations
from src.hybrid_search import hybrid_search

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "questions.jsonl"
LABELS = ROOT / "eval" / "labels"
LABELS.mkdir(parents=True, exist_ok=True)

ANSWERS = LABELS / "answers.jsonl"
POOL_CONFLICTS = LABELS / "_pool_conflicts.jsonl"
POOL_GROUNDING = LABELS / "_pool_grounding.jsonl"


def pair_id(text_a: str, text_b: str) -> str:
    """Stable id for a chunk pair, order-independent.

    Keyed on content, not on the `chunks.id` bigserial: re-running index.py
    reassigns those, which would orphan every label in the set.
    """
    a, b = sorted([text_a.strip(), text_b.strip()])
    return hashlib.sha1(f"{a}\x00{b}".encode()).hexdigest()[:12]


def sentence_id(question: str, sentence: str) -> str:
    """Stable id for a labelled sentence, keyed on the question it came from
    (the same sentence can appear under two questions and be grounded in one)."""
    return hashlib.sha1(f"{question.strip()}\x00{sentence.strip()}".encode()).hexdigest()[:12]


def load_questions(limit: int | None = None, category: str | None = None,
                   ids: list[int] | None = None) -> list[dict]:
    rows = [json.loads(line) for line in QUESTIONS.open(encoding="utf-8") if line.strip()]
    if ids:
        by_id = {q["id"]: q for q in rows}
        return [by_id[i] for i in ids if i in by_id]
    if category:
        rows = [q for q in rows if q["category"] == category]
    return rows[:limit] if limit else rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


# --- conflict pool ----------------------------------------------------------

def harvest_conflicts(questions: list[dict]) -> list[dict]:
    """Every chunk pair the reranker puts in one context window, scored.

    This is the population the detector actually runs on, so it is the
    population its precision should be measured over. Pairs are deduped
    across questions — the same two chunks are retrieved together for
    several questions and should not be labelled (or counted) twice.
    """
    seen: dict[str, dict] = {}

    for q in questions:
        chunks = rerank(q["question"])
        if len(chunks) < 2:
            continue

        idx_pairs = list(permutations(range(len(chunks)), 2))
        scores = nli_score([(chunks[i]["text"], chunks[j]["text"]) for i, j in idx_pairs])

        # Mirror conflict.py exactly: collapse both directions to the WEAKER
        # contradiction. Also keep the stronger direction, so calibration can
        # revisit that decision on labelled data instead of by anecdote.
        both: dict[tuple[int, int], list[float]] = {}
        for (i, j), s in zip(idx_pairs, scores):
            both.setdefault((min(i, j), max(i, j)), []).append(s["contradiction"])

        for (i, j), values in both.items():
            a, b = chunks[i], chunks[j]
            pid = pair_id(a["text"], b["text"])
            if pid in seen:
                seen[pid]["queries"].append(q["question"])
                continue
            seen[pid] = {
                "pid": pid,
                "queries": [q["question"]],
                "a_source": a["source"], "a_section": a.get("section"), "a_text": a["text"],
                "b_source": b["source"], "b_section": b.get("section"), "b_text": b["text"],
                "contradiction_min": round(min(values), 4),
                "contradiction_max": round(max(values), 4),
                "same_source": a["source"] == b["source"],
            }

        print(f"  [{q['id']:3d}] {len(both):2d} pairs  {q['question'][:50]}")

    return sorted(seen.values(), key=lambda r: r["contradiction_min"], reverse=True)


# --- grounding pool ---------------------------------------------------------

def freeze_answers(questions: list[dict]) -> list[dict]:
    """Run the pipeline once and persist the answers verbatim."""
    rows = []
    for q in questions:
        result = answer_question(q["question"], scopes=UNRESTRICTED)
        if result["refused"]:
            print(f"  [{q['id']:3d}] refused — skipped")
            continue
        rows.append({
            "qid": q["id"],
            "question": q["question"],
            "answer": result["answer"],
            # Context chunks in label order: chunks[0] is what the answer
            # cites as [1]. Grounding labels are meaningless without them.
            "chunks": [{"source": c["source"], "section": c.get("section"), "text": c["text"]}
                       for c in result["chunks"]],
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": "llama3.1:8b",
            "temperature": 0.2,
        })
        n = len(split_sentences(result["answer"]))
        print(f"  [{q['id']:3d}] {n:2d} sentences  {q['question'][:50]}")
    return rows


def harvest_grounding(answers: list[dict]) -> list[dict]:
    """Split frozen answers into sentences and attach the premises verify.py
    would score them against, plus those scores.

    Re-retrieval here has to match verify.py's, otherwise the calibrated
    threshold is calibrated against premises production never sees.
    """
    rows = []
    for row in answers:
        chunks = row["chunks"]
        for sentence in split_sentences(row["answer"]):
            claim = strip_citations(sentence)
            labels = cited_labels(sentence)

            if not _is_claim(sentence):
                # verify.py tags these `skipped` without scoring. They are kept
                # in the pool (marked) so the labeller can check that the
                # is-it-a-claim filter is dropping the right things.
                rows.append({
                    "sid": sentence_id(row["question"], sentence),
                    "qid": row["qid"], "question": row["question"],
                    "sentence": sentence, "claim": claim, "cited": labels,
                    "skipped_by_code": True, "premises": [],
                    "entailment": None, "contradiction": None,
                })
                continue

            premises = [chunks[n - 1] for n in labels if 1 <= n <= len(chunks)]
            premises += hybrid_search(claim, top_n=RETRIEVE_K)

            seen, unique = set(), []
            for p in premises:
                key = p["text"][:200]
                if key not in seen:
                    seen.add(key)
                    unique.append(p)

            scores = nli_score([(p["text"], claim) for p in unique])
            best = max(range(len(scores)), key=lambda i: scores[i]["entailment"])

            rows.append({
                "sid": sentence_id(row["question"], sentence),
                "qid": row["qid"], "question": row["question"],
                "sentence": sentence, "claim": claim, "cited": labels,
                "skipped_by_code": False,
                "premises": [{"source": p["source"], "section": p.get("section"),
                              "text": p["text"],
                              "entailment": round(s["entailment"], 4),
                              "contradiction": round(s["contradiction"], 4)}
                             for p, s in zip(unique, scores)],
                "best_premise": best,
                "entailment": round(scores[best]["entailment"], 4),
                "contradiction": round(max(s["contradiction"] for s in scores), 4),
            })
        print(f"  [{row['qid']:3d}] {row['question'][:50]}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["conflicts", "answers"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--category")
    ap.add_argument("--ids", help="comma-separated question ids, e.g. 1,3,8")
    ap.add_argument("--regenerate", action="store_true",
                    help="re-run generation even though answers.jsonl exists "
                         "(INVALIDATES every grounding label)")
    args = ap.parse_args()

    ids = [int(n) for n in args.ids.split(",")] if args.ids else None
    questions = load_questions(args.limit, args.category, ids)

    if args.what == "conflicts":
        print(f"Harvesting chunk pairs from {len(questions)} questions...")
        rows = harvest_conflicts(questions)
        write_jsonl(POOL_CONFLICTS, rows)
        flagged = sum(1 for r in rows if r["contradiction_min"] >= 0.60)
        print(f"\n{len(rows)} unique pairs -> {POOL_CONFLICTS}")
        print(f"{flagged} above the current CONTRADICTION_THRESHOLD of 0.60")
        return

    existing = read_jsonl(ANSWERS)
    if existing and not args.regenerate:
        print(f"{ANSWERS} already holds {len(existing)} answers — reusing them.\n"
              f"Grounding labels are attached to these exact sentences; pass "
              f"--regenerate only if you intend to discard them.")
        answers = existing
    else:
        if existing:
            print(f"REGENERATING over {len(existing)} frozen answers. Any label in "
                  f"grounding.jsonl whose sentence is not reproduced becomes orphaned "
                  f"(calibrate.py will report them).\n")
        print(f"Generating answers for {len(questions)} questions "
              f"(~1 min each on CPU)...")
        answers = freeze_answers(questions)
        write_jsonl(ANSWERS, answers)
        print(f"\n{len(answers)} answers -> {ANSWERS}")

    print("\nScoring sentences against their premises...")
    rows = harvest_grounding(answers)
    write_jsonl(POOL_GROUNDING, rows)
    claims = [r for r in rows if not r["skipped_by_code"]]
    print(f"\n{len(rows)} sentences ({len(claims)} scorable) -> {POOL_GROUNDING}")


if __name__ == "__main__":
    main()
