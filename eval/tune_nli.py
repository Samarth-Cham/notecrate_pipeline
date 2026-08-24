"""
Price the verification pass's speed/accuracy trade-off.

  python eval/tune_nli.py
  python eval/tune_nli.py --models current,minilm --premises 3,2,1

After the reranker swap, `verify` is ~60% of end-to-end work (6.39s avg,
15.76s p95 under load) — it is now the bottleneck that reranking used to be.
Two knobs move it, and both trade accuracy for speed:

  NLI_MODEL     the cross-encoder scoring (premise, claim) pairs
  RETRIEVE_K    how many premises each sentence is scored against

This replays eval/labels/grounding.jsonl — 54 sentences with their labels and
the premises that were actually retrieved for them — so accuracy is measured
against the same rubric verify.py is judged by, not re-derived.

Worth knowing before reading the output: the current configuration scores
~0.54 tag accuracy against a ~0.63 sweep ceiling, so there is not much
accuracy to protect. A faster model that holds ~0.54 is a straight win; the
question is whether one does.
"""

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.nli as nli_mod
from src.verify import (
    CONTRADICTED_THRESHOLD,
    GROUNDED_THRESHOLD,
    INFERRED_THRESHOLD,
)

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "eval" / "labels" / "grounding.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODELS = {
    "current":      "cross-encoder/nli-deberta-v3-base",       # ~184M
    "distilroberta": "cross-encoder/nli-distilroberta-base",   # ~82M
    "minilm":       "cross-encoder/nli-MiniLM2-L6-H768",       # ~66M
}

TAGS = ("grounded", "inferred", "uncertain")


def tag_for(scores: list[dict]) -> str:
    """verify.py's tagging rule, kept in step with it deliberately.

    Contradiction is read from the best-entailing premise rather than a max
    over all of them — an off-topic chunk's objection is not evidence.
    """
    best = max(range(len(scores)), key=lambda i: scores[i]["entailment"])
    entail = scores[best]["entailment"]
    contra = scores[best]["contradiction"]

    if contra >= CONTRADICTED_THRESHOLD and contra > entail:
        return "uncertain"
    if entail >= GROUNDED_THRESHOLD:
        return "grounded"
    if entail >= INFERRED_THRESHOLD:
        return "inferred"
    return "uncertain"


def macro_f1(pairs: list[tuple[str, str]]) -> float:
    """Unweighted mean F1. Macro, not micro: `inferred` is 7 of 54 rows, so a
    model that never predicts it would still look fine on plain accuracy."""
    f1s = []
    for tag in TAGS:
        tp = sum(1 for t, p in pairs if t == tag and p == tag)
        fp = sum(1 for t, p in pairs if t != tag and p == tag)
        fn = sum(1 for t, p in pairs if t == tag and p != tag)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall)
                   if precision + recall else 0.0)
    return sum(f1s) / len(f1s)


def evaluate(rows: list[dict], max_premises: int) -> dict:
    pairs, latencies = [], []

    for row in rows:
        premises = [p["text"] for p in row["premises"]][:max_premises]
        if not premises:
            continue
        started = time.perf_counter()
        scores = nli_mod.score([(p, row["claim"]) for p in premises])
        latencies.append(time.perf_counter() - started)
        pairs.append((row["label"], tag_for(scores)))

    correct = sum(1 for t, p in pairs if t == p)
    return {
        "accuracy": correct / len(pairs),
        "macro_f1": macro_f1(pairs),
        "per_sentence_s": statistics.mean(latencies),
        "predicted": dict(Counter(p for _, p in pairs)),
        "n": len(pairs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--premises", default="3,2,1",
                    help="premise counts to try per sentence")
    args = ap.parse_args()

    rows = [json.loads(line) for line in LABELS.open(encoding="utf-8")
            if line.strip()]
    counts = [int(n) for n in args.premises.split(",")]

    results, baseline = {}, None
    print(f"{len(rows)} labelled sentences "
          f"({dict(Counter(r['label'] for r in rows))})\n")
    print(f"{'model':15s} {'prem':>4s} {'acc':>7s} {'macroF1':>8s} "
          f"{'per-sent':>9s} {'speedup':>8s}")
    print("-" * 60)

    for name in args.models.split(","):
        name = name.strip()
        if name not in MODELS:
            print(f"unknown model {name!r}; known: {list(MODELS)}")
            continue

        nli_mod.NLI_MODEL = MODELS[name]
        nli_mod._model = None
        try:
            nli_mod._load()
        except Exception as e:
            print(f"{name:15s} FAILED to load: {str(e)[:60]}")
            continue

        for n_prem in counts:
            r = evaluate(rows, n_prem)
            key = f"{name}/{n_prem}"
            results[key] = {**r, "model": MODELS[name], "premises": n_prem}
            if baseline is None:
                baseline = r["per_sentence_s"]
            speed = f"{baseline / r['per_sentence_s']:.1f}x"
            print(f"{name:15s} {n_prem:4d} {r['accuracy']:7.3f} "
                  f"{r['macro_f1']:8.3f} {r['per_sentence_s']:8.3f}s {speed:>8s}")

    if not results:
        return
    out = RESULTS_DIR / f"tune_nli_{datetime.now():%Y%m%d_%H%M}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")
    print("\nNote: `predicted` counts are in the JSON. A config that never "
          "predicts `inferred` can still score well on accuracy — check them "
          "before trusting a winner.")


if __name__ == "__main__":
    main()
