"""
Price the reranker speed/quality trade-off against the eval set.

  python eval/tune_rerank.py
  python eval/tune_rerank.py --configs current,minilm

Week 5 asks for "before/after tuning numbers". The tuning knobs that matter
are the reranker MODEL and how many CANDIDATES it scores, because together
they are ~56% of end-to-end latency on the 2-core staging VM — and both trade
retrieval quality for speed.

The point is to make that trade visible instead of guessing. A 22M model is
7.8x faster than the 278M one on CPU; whether it retrieves as well is a
question about this corpus, not a general one.

Latency here is measured on the machine you run it on, so treat the ratios as
transferable and the absolute numbers as not.
"""

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.rerank as rerank_mod
from src.permissions import UNRESTRICTED

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "questions.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BASE = "BAAI/bge-reranker-base"
MINI = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CONFIGS = {
    "current":    (BASE, 20),
    "cand10":     (BASE, 10),
    "minilm":     (MINI, 20),
    "minilm10":   (MINI, 10),
}


def apply(model: str, candidates: int) -> None:
    """Swap the reranker configuration in place.

    _model must be cleared or _load() returns the previously cached model and
    every config after the first would silently measure the first one.
    """
    rerank_mod.RERANK_MODEL = model
    rerank_mod.CANDIDATES = candidates
    rerank_mod._model = None
    rerank_mod._load()          # pay the load cost before timing anything


def score(questions: list[dict]) -> dict:
    hits, precisions, recalls, rrs, latencies = [], [], [], [], []

    for q in questions:
        expected = set(q["expected_sources"])
        if not expected:
            continue            # negative questions have no retrieval metrics

        started = time.perf_counter()
        results = rerank_mod.rerank(q["question"], scopes=UNRESTRICTED)
        latencies.append(time.perf_counter() - started)

        sources = [r["source"] for r in results]
        flags = [s in expected for s in sources]

        hits.append(any(flags))
        precisions.append(sum(flags) / len(flags) if flags else 0.0)
        recalls.append(len(expected & set(sources)) / len(expected))
        rrs.append(next((1 / (i + 1) for i, f in enumerate(flags) if f), 0.0))

    return {
        "hit_rate": sum(hits) / len(hits),
        "precision_at_5": sum(precisions) / len(precisions),
        "recall_at_5": sum(recalls) / len(recalls),
        "mrr": sum(rrs) / len(rrs),
        "latency_mean_s": statistics.mean(latencies),
        "latency_p95_s": sorted(latencies)[min(len(latencies) - 1,
                                               int(0.95 * len(latencies)))],
        "n": len(hits),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS))
    args = ap.parse_args()

    questions = [json.loads(line) for line in QUESTIONS.open(encoding="utf-8")
                 if line.strip()]

    rows = {}
    for name in args.configs.split(","):
        name = name.strip()
        if name not in CONFIGS:
            print(f"unknown config {name!r}; known: {list(CONFIGS)}")
            continue
        model, candidates = CONFIGS[name]
        print(f"\n=== {name}: {model}, {candidates} candidates ===", flush=True)
        apply(model, candidates)
        rows[name] = {**score(questions), "model": model, "candidates": candidates}
        r = rows[name]
        print(f"  hit@5 {r['hit_rate']:.3f}  P@5 {r['precision_at_5']:.3f}  "
              f"R@5 {r['recall_at_5']:.3f}  MRR {r['mrr']:.3f}  "
              f"rerank {r['latency_mean_s']:.2f}s", flush=True)

    if not rows:
        return

    print(f"\n{'=' * 82}")
    print(f"{'config':12s} {'model':10s} {'cand':>5s} {'hit@5':>7s} {'P@5':>7s} "
          f"{'R@5':>7s} {'MRR':>7s} {'rerank':>8s} {'speedup':>8s}")
    print("-" * 82)
    base_latency = rows.get("current", {}).get("latency_mean_s")
    for name, r in rows.items():
        short = "MiniLM" if r["model"] == MINI else "bge-base"
        speed = f"{base_latency / r['latency_mean_s']:.1f}x" if base_latency else "-"
        print(f"{name:12s} {short:10s} {r['candidates']:5d} {r['hit_rate']:7.3f} "
              f"{r['precision_at_5']:7.3f} {r['recall_at_5']:7.3f} {r['mrr']:7.3f} "
              f"{r['latency_mean_s']:7.2f}s {speed:>8s}")

    out = RESULTS_DIR / f"tune_rerank_{datetime.now():%Y%m%d_%H%M}.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
