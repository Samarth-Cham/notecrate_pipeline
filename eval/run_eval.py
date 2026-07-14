"""
Score retrieval quality against eval/questions.jsonl.

  python eval/run_eval.py            # run all questions, print + save results

Metrics (per the NoteCrate plan, Workstream C):
  hit@k       — did ANY expected source appear in the top-k?
  precision@k — fraction of retrieved chunks that are from expected sources
  recall@k    — fraction of expected sources that appeared at all
  MRR         — 1/rank of the first relevant hit, averaged
  refusal     — for negative/freshness questions: did top score stay under the floor?
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import psycopg
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

OLLAMA = os.environ["OLLAMA_URL"]
EMBED_MODEL = os.environ["EMBED_MODEL"]
DB_URL = os.environ["DATABASE_URL"]

TOP_K = 5
NOISE_FLOOR = 0.57

QUESTIONS = ROOT / "eval" / "questions.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def embed(text: str) -> list[float]:
    r = requests.post(f"{OLLAMA}/api/embeddings",
                      json={"model": EMBED_MODEL, "prompt": text})
    r.raise_for_status()
    return r.json()["embedding"]


def retrieve(query: str) -> list[dict]:
    qvec = embed(query)
    with psycopg.connect(DB_URL) as conn:
        rows = conn.execute("""
            SELECT source, 1 - (embedding <=> %s::vector) AS score
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (str(qvec), str(qvec), TOP_K)).fetchall()
    return [{"source": r[0], "score": r[1]} for r in rows]


def score_question(q: dict) -> dict:
    hits = retrieve(q["question"])
    retrieved_sources = [h["source"] for h in hits]
    top_score = hits[0]["score"] if hits else 0.0
    expected = set(q["expected_sources"])

    result = {"id": q["id"], "category": q["category"],
              "question": q["question"], "top_score": round(top_score, 3)}

    if not expected:
        # negative / freshness question: success = staying under the floor
        result["should_refuse"] = True
        result["refused_correctly"] = top_score < NOISE_FLOOR
        return result

    # positive question: standard retrieval metrics
    relevant_flags = [s in expected for s in retrieved_sources]

    result["should_refuse"] = False
    result["hit_at_k"] = any(relevant_flags)
    result["precision_at_k"] = round(sum(relevant_flags) / len(relevant_flags), 3) if relevant_flags else 0.0
    result["recall_at_k"] = round(len(expected & set(retrieved_sources)) / len(expected), 3)
    # MRR contribution: 1/rank of first relevant result, 0 if none
    result["reciprocal_rank"] = round(
        next((1 / (i + 1) for i, rel in enumerate(relevant_flags) if rel), 0.0), 3)
    result["retrieved"] = retrieved_sources
    return result


def main():
    questions = [json.loads(l) for l in QUESTIONS.open(encoding="utf-8") if l.strip()]
    results = [score_question(q) for q in questions]

    positives = [r for r in results if not r["should_refuse"]]
    negatives = [r for r in results if r["should_refuse"]]

    print(f"\n{'='*70}\nEVAL: {len(questions)} questions "
          f"({len(positives)} positive, {len(negatives)} negative/freshness)\n{'='*70}")

    for r in results:
        if r["should_refuse"]:
            mark = "PASS" if r["refused_correctly"] else "FAIL"
            print(f"[{r['id']:3d}] {mark}  refuse-check   top={r['top_score']}  {r['question'][:50]}")
        else:
            mark = "PASS" if r["hit_at_k"] else "FAIL"
            print(f"[{r['id']:3d}] {mark}  P@5={r['precision_at_k']:.2f} "
                  f"R@5={r['recall_at_k']:.2f} RR={r['reciprocal_rank']:.2f}  {r['question'][:50]}")

    # aggregates
    if positives:
        agg = {
            "hit_rate": sum(r["hit_at_k"] for r in positives) / len(positives),
            "mean_precision_at_k": sum(r["precision_at_k"] for r in positives) / len(positives),
            "mean_recall_at_k": sum(r["recall_at_k"] for r in positives) / len(positives),
            "mrr": sum(r["reciprocal_rank"] for r in positives) / len(positives),
        }
    else:
        agg = {}
    refusal_rate = (sum(r["refused_correctly"] for r in negatives) / len(negatives)) if negatives else None

    print(f"\n{'-'*70}\nAGGREGATES (positive questions):")
    for k, v in agg.items():
        print(f"  {k:22s} {v:.3f}")
    if refusal_rate is not None:
        print(f"  {'refusal_accuracy':22s} {refusal_rate:.3f}")

    # per-category slice — this is where corpus-balance effects show up
    print(f"\nBY CATEGORY (hit rate):")
    cats = sorted({r["category"] for r in positives})
    for cat in cats:
        sub = [r for r in positives if r["category"] == cat]
        print(f"  {cat:10s} {sum(r['hit_at_k'] for r in sub)}/{len(sub)}")

    # save timestamped snapshot
    label = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    out = RESULTS_DIR / f"{label}_{datetime.now():%Y%m%d_%H%M}.json"
    out.write_text(json.dumps({"label": label, "aggregates": agg,
                               "refusal_accuracy": refusal_rate,
                               "results": results}, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()