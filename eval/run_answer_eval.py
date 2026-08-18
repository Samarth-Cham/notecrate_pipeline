"""
Generation-side eval: score the ANSWERS, not just the retrieved chunks.

  python eval/run_answer_eval.py                  # all questions
  python eval/run_answer_eval.py --limit 5        # smoke run
  python eval/run_answer_eval.py --category k8s
  python eval/run_answer_eval.py --no-verify      # skip grounding tags (much faster)

run_eval.py answers "did we retrieve the right chunks". This answers "was
the answer any good", which is what Week 4's features actually change.

Metrics — these are local reimplementations of the RAGAS definitions rather
than the `ragas` package. Same formulas, but scored with the Llama 3.1 and
embedding models already running for the pipeline, so the harness needs no
hosted LLM key and runs fully offline. Swapping in the library later means
replacing this file, not the pipeline.

  faithfulness      fraction of answer sentences the verification pass tags
                    `grounded` (RAGAS decomposes into statements and asks an
                    LLM if the context supports each; our NLI pass does the
                    same job with a smaller, cheaper model)
  answer_relevancy  the model writes questions the answer would answer;
                    mean cosine similarity of those to the real question
  keyword_coverage  fraction of `expected_answer_contains` terms present —
                    deterministic, no model in the loop, catches regressions
                    the LLM-scored metrics are too noisy to see
  refusal           negative/freshness questions: did the pipeline decline, by
                    EITHER mechanism (see below)?

Refusal has two layers and they catch different things:

  the gate   cosine similarity below the noise floor, so generation never runs.
             Cheap (~11s vs ~60s) but it can only see topical distance.
  the model  the answer itself declines after reading the sources.

A freshness question — "how did we fix X", where X happened after the corpus
was exported — is topically ON-corpus, so the gate cannot catch it by
construction. Only a model that has read the sources can notice the answer
isn't there. Scoring the gate alone therefore understates the system: it marks
a correct, graceful refusal as a failure.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm import chat_json, embed
from src.pipeline import answer_question
from src.verify import grounding_summary

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "questions.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RELEVANCY_N = 2   # questions generated per answer

RELEVANCY_SYSTEM = (
    "Given an answer, write the questions it was most likely responding to. "
    'Reply with JSON only: {"questions": ["...", "..."]}. '
    f"Write exactly {RELEVANCY_N} questions. Do not explain."
)

REFUSAL_SYSTEM = (
    "You are grading whether an answer DECLINES to answer the question.\n"
    'Reply with JSON only: {"declines": true} or {"declines": false}.\n'
    "declines = true when the answer says the information is not in its "
    "sources, or that the question's premise is wrong.\n"
    "declines = false when the answer actually addresses the question, even "
    "if it adds caveats or hedges."
)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def answer_relevancy(question: str, answer: str) -> float | None:
    """Reverse-generate questions from the answer, compare to the real one.

    An answer that drifts off-topic produces questions unlike the original
    even when every sentence in it is perfectly grounded — this is the
    metric that catches "true but not what was asked".
    """
    parsed = chat_json([
        {"role": "system", "content": RELEVANCY_SYSTEM},
        {"role": "user", "content": answer},
    ])
    if not isinstance(parsed, dict):
        return None
    generated = [q for q in parsed.get("questions", []) if isinstance(q, str) and q.strip()]
    if not generated:
        return None

    qvec = embed(question)
    return sum(cosine(qvec, embed(g)) for g in generated) / len(generated)


def declines_to_answer(question: str, answer: str) -> bool:
    """Did the generated answer decline? Judged by the LLM rather than by
    phrase-matching: a graceful refusal reads like "we didn't actually fix
    that; the sources discuss GitHub auth, not pgvector", which shares no
    fixed wording with "there is no information on this in the sources"."""
    parsed = chat_json([
        {"role": "system", "content": REFUSAL_SYSTEM},
        {"role": "user", "content": f"Question: {question}\n\nAnswer: {answer}"},
    ])
    return bool(parsed.get("declines")) if isinstance(parsed, dict) else False


def keyword_coverage(answer: str, expected: list[str]) -> float | None:
    if not expected:
        return None
    low = answer.lower()
    return sum(1 for term in expected if term.lower() in low) / len(expected)


def score_question(q: dict, *, verify: bool, conflicts: bool = False) -> dict:
    started = time.perf_counter()
    result = answer_question(q["question"], verify_answer=verify,
                             detect_conflicts_enabled=conflicts)
    elapsed = time.perf_counter() - started

    row = {
        "id": q["id"],
        "category": q["category"],
        "question": q["question"],
        "route_kind": result["route"]["kind"],
        "gate_refused": result["refused"],
        "should_refuse": not q["expected_sources"],
        "vector_top_score": result["vector_top_score"],   # for floor calibration
        "latency_s": round(elapsed, 1),
        "conflicts": len(result["conflicts"]),
    }

    if result["refused"]:
        # Generation never ran. Correct only if this was a negative question.
        row["refused_correctly"] = row["should_refuse"]
        return row

    row["answer"] = result["answer"]

    if row["should_refuse"]:
        # The gate let it through, but the model still gets to decline after
        # reading the sources — the only layer that can catch a freshness gap.
        row["answer_refused"] = declines_to_answer(q["question"], result["answer"])
        row["refused_correctly"] = row["answer_refused"]
        return row

    row["confidence"] = result["confidence"]
    row["keyword_coverage"] = keyword_coverage(
        result["answer"], q.get("expected_answer_contains", []))
    row["answer_relevancy"] = answer_relevancy(q["question"], result["answer"])

    if result["sentences"]:
        summary = grounding_summary(result["sentences"])
        row["faithfulness"] = summary["faithfulness"]
        row["grounding"] = summary

    return row


def mean(values: list) -> float | None:
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 3) if clean else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--category")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the verification pass (drops faithfulness)")
    ap.add_argument("--label", default="answers")
    ap.add_argument("--conflicts", action="store_true",
                    help="re-enable conflict detection (off in the pipeline by "
                         "default; use this to measure its cost)")
    args = ap.parse_args()

    questions = [json.loads(line) for line in QUESTIONS.open(encoding="utf-8") if line.strip()]
    if args.category:
        questions = [q for q in questions if q["category"] == args.category]
    if args.limit:
        questions = questions[:args.limit]

    print(f"\n{'=' * 78}\nANSWER EVAL: {len(questions)} questions"
          f"{' (no verification pass)' if args.no_verify else ''}\n{'=' * 78}")

    rows = []
    for q in questions:
        row = score_question(q, verify=not args.no_verify, conflicts=args.conflicts)
        rows.append(row)

        if row["should_refuse"] or row["gate_refused"]:
            mark = "PASS" if row.get("refused_correctly") else "FAIL"
            state = ("refused@gate" if row["gate_refused"]
                     else "refused@model" if row.get("answer_refused")
                     else "ANSWERED")
            print(f"[{row['id']:3d}] {mark}  {state:13s} "
                  f"vec={row['vector_top_score']:.3f} "
                  f"{row['latency_s']:5.1f}s  {row['question'][:44]}")
        else:
            print(f"[{row['id']:3d}] faith={_fmt(row.get('faithfulness'))} "
                  f"rel={_fmt(row.get('answer_relevancy'))} "
                  f"kw={_fmt(row.get('keyword_coverage'))} "
                  f"{row['route_kind']:9s} {row['latency_s']:5.1f}s  "
                  f"{row['question'][:44]}")

    answered = [r for r in rows if not r["gate_refused"] and not r["should_refuse"]]
    negatives = [r for r in rows if r["should_refuse"]]

    agg = {
        "faithfulness": mean([r.get("faithfulness") for r in answered]),
        "answer_relevancy": mean([r.get("answer_relevancy") for r in answered]),
        "keyword_coverage": mean([r.get("keyword_coverage") for r in answered]),
        "latency_s_mean": mean([r["latency_s"] for r in rows]),
        "latency_s_p95": _p95([r["latency_s"] for r in rows]),
        "answered": len(answered),
        "multi_hop_routed": sum(1 for r in rows if r["route_kind"] == "multi_hop"),
        "conflicts_flagged": sum(r["conflicts"] for r in rows),
    }
    if negatives:
        agg["refusal_accuracy"] = mean([float(r.get("refused_correctly", False))
                                        for r in negatives])
        # Split by mechanism: the gate is ~5x faster because generation never
        # runs, so a shift from gate to model is a latency regression even when
        # overall accuracy holds.
        agg["refused_at_gate"] = sum(1 for r in negatives if r["gate_refused"])
        agg["refused_at_model"] = sum(1 for r in negatives if r.get("answer_refused"))

    print(f"\n{'-' * 78}\nAGGREGATES:")
    for k, v in agg.items():
        print(f"  {k:22s} {v}")

    out = RESULTS_DIR / f"{args.label}_{datetime.now():%Y%m%d_%H%M}.json"
    out.write_text(json.dumps({"label": args.label, "verified": not args.no_verify,
                               "aggregates": agg, "results": rows}, indent=2),
                   encoding="utf-8")
    print(f"\nSaved -> {out}")


def _fmt(v) -> str:
    return f"{v:.2f}" if isinstance(v, (int, float)) else " -- "


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 1)


if __name__ == "__main__":
    main()
