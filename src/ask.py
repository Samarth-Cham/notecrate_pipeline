"""
Ask a question, get a cited answer from the corpus.

  python src/ask.py "how does pod restart policy work"
  python src/ask.py "how does pod restart policy work" --no-verify
  python src/ask.py "how do I set up React" --role=senior
  python src/ask.py "what about the other one?" --conversation=demo

Thin CLI over src/pipeline.py — see that module for the actual flow.
Sentences are prefixed with their grounding tag:

  [G] grounded   [I] inferred   [?] uncertain
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import answer_question
from src.verify import grounding_summary

TAG_MARK = {"grounded": "[G]", "inferred": "[I]", "uncertain": "[?]", "skipped": "   "}


def _flag_value(flags: set[str], name: str) -> str | None:
    prefix = f"--{name}="
    return next((f[len(prefix):] for f in flags if f.startswith(prefix)), None)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}

    query = args[0] if args else "how does pod restart policy work"
    result = answer_question(
        query,
        verify_answer="--no-verify" not in flags,
        role=_flag_value(flags, "role"),
        conversation_id=_flag_value(flags, "conversation"),
    )

    print(f"\nQ: {query}\n" + "=" * 60)

    if result["history"]:
        print("Recalled from this conversation (by relevance, not recency):")
        for h in result["history"]:
            rel = f"{h['relevance']:.2f}" if h["relevance"] is not None else "recent"
            print(f"  ({rel}) {h['question'][:60]}")
        print()

    if result["refused"]:
        print(f"\n{result['reason']} Not asking the model.")
        return

    if result["route"]["kind"] == "multi_hop":
        print("Router: multi-hop, decomposed into")
        for s in result["route"]["sub_queries"]:
            print(f"  - {s}")
        print()

    if result["confidence"] == "weak":
        print(f"[weak retrieval: best score {result['vector_top_score']} "
              f"- answer may be thin]\n")

    if result["sentences"]:
        for s in result["sentences"]:
            print(f"{TAG_MARK[s['tag']]} {s['text']}")
        print()
        summary = grounding_summary(result["sentences"])
        print(f"Grounding: {summary['grounded']} grounded, "
              f"{summary['inferred']} inferred, {summary['uncertain']} uncertain "
              f"(faithfulness {summary['faithfulness']})")
    else:
        print(result["answer"])

    if result["conflicts"]:
        print("\n" + "-" * 60 + "\nSources disagree:")
        for f in result["conflicts"]:
            a, b = f["a"] + 1, f["b"] + 1
            print(f"  [{a}] {f['a_source']}  vs  [{b}] {f['b_source']}  "
                  f"(p={f['score']:.2f})")

    role_note = f"  (role: {result['role']})" if result["role"] else ""
    print("\n" + "-" * 60 + f"\nSources:{role_note}")
    for i, c in enumerate(result["chunks"], 1):
        sec = f" - {c['section']}" if c.get("section") else ""
        tags = "+".join(c["roles"]) if c.get("roles") else "-"
        # Show the role adjustment when one was applied, so the effect is visible.
        delta = c["rerank_score"] - c["relevance_score"]
        adj = f" {delta:+.2f}" if abs(delta) > 1e-9 else "     "
        print(f"  [{i}] ({c['rerank_score']:+.2f}{adj}) [{tags}] {c['source']}{sec}")


if __name__ == "__main__":
    main()
