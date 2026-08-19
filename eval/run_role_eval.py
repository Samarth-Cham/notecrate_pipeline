"""
Role-aware retrieval: same query, both roles, side by side.

  python eval/run_role_eval.py                  # retrieval only (fast)
  python eval/run_role_eval.py --answers        # also generate both answers

Plan section 7 scores this feature by "qualitative side-by-side review" — the
team looks at one query answered for two roles and judges whether depth and
framing differ appropriately. There is no ground truth to score against, so
this script's job is to lay the comparison out, not to grade it.

It does report one hard number: retrieval OVERLAP between the two roles. That
is the check that the feature is a boost and not a filter. Overlap of 0 means
the roles have been given disjoint corpora, which is the failure mode section
2.3 explicitly warns against; overlap of 1.0 across every query means the
boost is too weak to do anything.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.permissions import UNRESTRICTED
from src.pipeline import MAX_CONTEXT, _merge, answer_question
from src.rerank import rerank
from src.roles import ROLES

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Queries where the corpus plausibly holds both introductory and deep material.
QUERIES = [
    "how do I set up a github repo",
    "how does garbage collection work",
    "how do taints and tolerations work",
    "how do I set up React on Windows",
    "how does the scheduler assign pods to nodes",
    "what is a ConfigMap",
    "how does quick sort work",
    "how do init containers work",
]


def retrieve_for(query: str, role: str | None) -> list[dict]:
    return _merge([rerank(query, role=role)], MAX_CONTEXT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", action="store_true",
                    help="also generate an answer per role (slow)")
    args = ap.parse_args()

    rows = []
    for query in QUERIES:
        print("=" * 78)
        print(f"Q: {query}")

        per_role = {}
        for role in (None, *ROLES):
            chunks = retrieve_for(query, role)
            per_role[role or "none"] = chunks
            label = role or "no role"
            print(f"\n  --- {label} ---")
            for c in chunks[:4]:
                tags = "+".join(c["roles"]) if c.get("roles") else "-"
                delta = c["rerank_score"] - c["relevance_score"]
                adj = f"{delta:+.2f}" if abs(delta) > 1e-9 else "    "
                print(f"    {c['rerank_score']:.3f} ({adj}) [{tags:6s}] "
                      f"{c['source'][:44]}")

        jr = [c["id"] for c in per_role["junior"]]
        sr = [c["id"] for c in per_role["senior"]]
        overlap = len(set(jr) & set(sr)) / len(set(jr) | set(sr)) if jr or sr else 0.0
        print(f"\n  junior/senior chunk overlap: {overlap:.2f}")

        row = {"query": query, "overlap": round(overlap, 3),
               "junior_sources": [c["source"] for c in per_role["junior"]],
               "senior_sources": [c["source"] for c in per_role["senior"]]}

        if args.answers:
            for role in ROLES:
                r = answer_question(query, verify_answer=False, role=role, scopes=UNRESTRICTED)
                row[f"{role}_answer"] = r["answer"]
                print(f"\n  [{role} answer] {(r['answer'] or '')[:260]}")

        rows.append(row)
        print()

    mean_overlap = sum(r["overlap"] for r in rows) / len(rows)
    identical = sum(1 for r in rows if r["overlap"] == 1.0)
    disjoint = sum(1 for r in rows if r["overlap"] == 0.0)

    print("-" * 78)
    print(f"mean junior/senior overlap : {mean_overlap:.3f}")
    print(f"queries where roles agree  : {identical}/{len(rows)}  (boost too weak)")
    print(f"queries fully disjoint     : {disjoint}/{len(rows)}  (acting as a filter)")

    out = RESULTS_DIR / f"roles_{datetime.now():%Y%m%d_%H%M}.json"
    out.write_text(json.dumps({"mean_overlap": round(mean_overlap, 3),
                               "results": rows}, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
