"""Regenerate eval/labels/grounding.jsonl from the pool + the labels below.

  python eval/harvest_labels.py answers          # rebuilds the pool first
  python eval/labels/build_grounding.py --force

Every label here was assigned by reading the sentence against the premises
stored in its pool row — NOT against the corpus at large, and not against what
is true of Kubernetes. See README.md for why that distinction decides what the
thresholds end up measuring.

grounding.jsonl is the source of truth once it exists; --force is required to
overwrite it so a rebuild cannot silently discard hand corrections.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
POOL = ROOT / "eval" / "labels" / "_pool_grounding.jsonl"
OUT = ROOT / "eval" / "labels" / "grounding.jsonl"

# sid -> (label, note, retrieval_miss)
LABELS = {
    # --- q1: pod restart policy ---------------------------------------------
    "4bd08189b30a": ("grounded", "The best premise contains this sentence almost verbatim. NLI scores it 0.10 because the SAME chunk also asserts the near-opposite in a bullet above (an inconsistency in the upstream Kubernetes doc), so the model sees a premise that both states and denies the claim.", False),
    "a2ef6d2c94d8": ("inferred", "No premise states the Pod-level rule outright. The sidecar premise shows a container-level restartPolicy in use, which supports the second half by example.", False),

    # --- q2: Deployment vs StatefulSet --------------------------------------
    "b19285818513": ("grounded", "statefulset.md lists exactly these four properties.", False),
    "79527b6f446f": ("grounded", "Both halves are stated: replicationcontroller.md for 'higher-level API, declarative, server-side, additional features', statefulset.md for the stateless recommendation. Scored 0.00 entailment anyway.", False),

    # --- q3: init vs sidecar containers -------------------------------------
    "b1d2babccec2": ("grounded", "Each conjunct is stated verbatim by a different premise.", False),
    "13760f0da761": ("grounded", "init-containers.md states this almost word for word.", False),
    "d9fec23b9035": ("grounded", "Both attributed claims are stated in their cited chunks. Note the stripped claim opens with a dangling 'claims' — SOURCE_VERB_RE only removes the attribution when followed by 'that'.", False),
    "5f4c1761745e": ("inferred", "The sidecar half is verbatim. 'Init containers ... cannot exchange messages with the app container' is not in any retrieved premise.", False),
    "b92f38ee2405": ("grounded", "sidecar-containers.md says 'Unlike init containers, sidecar containers support probes', which states both halves at once.", False),
    "62e864fbf366": ("uncertain", "Nothing retrieved says init containers share CPU/memory/network with app containers or that they do not interact directly.", True),

    # --- q5: HorizontalPodAutoscaler ----------------------------------------
    "4d5e411da505": ("grounded", "Premise: 'the HorizontalPodAutoscaler controller evaluates each metric, and proposes a new scale based on that metric'.", False),
    "1912fb483892": ("grounded", "Premise states the max-of-metrics rule verbatim.", False),
    "3a05a7b590b4": ("inferred", "The premises establish that the HPA scales a target from resource metrics; 'periodically' and the specific metric list are a reasonable synthesis, not stated.", False),

    # --- q6: least squares --------------------------------------------------
    "d6b93e8c1423": ("uncertain", "The claim is textbook-correct but the retrieved premises are a red-black-tree colouring, a permutations table and a homework sheet that poses the problem without stating the normal equations.", False),

    # --- q8: chunk size (the corpus disagrees with itself here) -------------
    "e08f7af463ae": ("uncertain", "The premises contain 900/150 as ANOTHER model's recommendation and separately record 'chunk size (600, settled for now)'. 'We chose 900/150' asserts a decision the premises do not support and one premise contradicts. The generator silently picked a side of a real corpus conflict.", False),
    "4a4315b26806": ("grounded", "Premise: 'That gives you enough context for technical topics without making each chunk too bulky.' Near-verbatim.", False),
    "5c233fedcbbe": ("grounded", "Premise states both halves: 'It is large enough to keep related concepts together. It is small enough to avoid stuffing too much context into a single retrieval unit.' Scored 0.00 because that text sits beyond MAX_CHARS=900 in a long chunk and is truncated away before the model sees it. No threshold can fix this one.", False),

    # --- q13: quicksort -----------------------------------------------------
    "8466983beb7e": ("inferred", "The premises demonstrate partitioning on worked examples but never state the general definition.", False),
    "77f2c6cbc355": ("grounded", "Premise: 'After the first partition, the pivot 5 is placed correctly.'", False),
    "8dd3a87bf860": ("grounded", "Premise enumerates exactly these three guarantees of the first pass.", False),
    "11a03ba15e4a": ("uncertain", "A claim about the corpus, not about quicksort. Nothing states that the sources disagree — and they do not; they are different arrays.", False),
    "a5be3e448c6c": ("uncertain", "The cited chunk lists [1,2,3,7,5] as one of four multiple-choice OPTIONS. The source does not claim it.", False),
    "e45cb4b2b1e4": ("grounded", "Premise: 'Resulting array after first pass: [3, 1, 4, 2, 5, 9, 8]. Correct answer.' The source does claim this.", False),
    "fd2ea00a3b68": ("uncertain", "Same as a5be3e448c6c: [1,3,2,4,9,6] is an unanswered multiple-choice option, not a claim.", False),
    "5774c8b8fdc3": ("uncertain", "Meta-claim about consistency. Its 0.99 entailment comes from a red-black-tree chunk saying 'this still doesn't match' — a spurious lexical match.", False),

    # --- q15: merge sort vs heap sort ---------------------------------------
    "5ba6ba2e8422": ("uncertain", "Flatly wrong and contradicted by the premise: the two-phase heapify/sort-down description belongs to HEAPSORT, not merge sort. The answer attributed heapsort's algorithm to merge sort.", False),

    # --- q19: hash tables ---------------------------------------------------
    "896df03e0d30": ("grounded", "pdf_20 states this verbatim.", False),
    "b6b24da521a0": ("inferred", "Determinism and index assignment are stated; the large-range-to-small-range framing is not.", False),

    # --- q20/q21: red-black trees -------------------------------------------
    "0ea8688138dd": ("grounded", "Premise states the black-height property verbatim.", False),
    "19007a96bdb0": ("grounded", "Premise states the root-absorbs-extra-black rule verbatim.", False),
    "1070a8e21eb6": ("inferred", "'Rotation needed when the uncle is black' is supported; 'recoloring works when the uncle is red' is the standard complement but is not stated in any premise.", False),
    "3f3c3e30b485": ("grounded", "Premises list the invariants (root black, no adjacent reds, equal black height) and state that the tester need not access private fields.", False),
    "ed6bfc498679": ("grounded", "Premise: 'Outputs: tree contains all inserted values / tree satisfies red-black properties.'", False),

    # --- q23: heap sort complexity ------------------------------------------
    "87f2998e80de": ("uncertain", "Meta-claim about disagreement, scored against gold-market chunks that retrieval surfaced by accident.", False),
    "cdf0f9519af0": ("grounded", "Premise states the per-removal cost decomposition this attributes to [1].", False),
    "ad0805c4c7c2": ("grounded", "Premise: 'If you start with a regular unsorted array, heap sort has two phases.'", False),
    "5a20fb467454": ("grounded", "The cited chunk does describe two phases and does not give a complexity — the sentence's claim about what the source omits is correct.", False),
    "e83df48fb870": ("grounded", "Premise states the bottom-up linearity argument verbatim.", False),
    "35b721d3aa99": ("grounded", "Premise states both cases are dominated by the extraction phase.", False),
    "92d53f2c673d": ("uncertain", "Meta-claim; its 0.56 entailment comes from an unrelated red-black-tree chunk.", False),

    # --- q26: clear-box vs I/O testing --------------------------------------
    "aee6905c7edb": ("grounded", "Premise states the definition and the 'can still interact only through public methods' caveat.", False),
    "143c18614ae2": ("grounded", "Premise: a clear-box test 'is designed using knowledge of' internals, but can interact through public methods only.", False),
    "d1c4ed2014d3": ("grounded", "Premise defines the I/O test in these terms.", False),
    "97d05b9e3c1e": ("grounded", "Premise: 'But the test can still interact only through public methods.'", False),
    "bb09e259142e": ("uncertain", "The first half is supported, but 'implying that clear-box and I/O tests are mutually exclusive' is contradicted by a premise that explicitly wants a test that is both.", False),
    "9591d8ae6c99": ("grounded", "Premise supports both the knowledge requirement and the no-private-fields caveat.", False),
    "f50fb879bf39": ("grounded", "Premise: reading internal state makes it 'a structural (white-box) test, not an input/output test'.", False),
    "d2143feb8d44": ("grounded", "Premise defines the I/O test exactly this way.", False),
    "e134b9f78936": ("grounded", "Premise: 'specific input -> specific output. So it is an I/O test. But it doesn't use internal knowledge.'", False),
    "cfa0a1b84a1d": ("uncertain", "Asserts a contradiction between sources that do not contradict; nothing in the premises supports it.", False),

    # --- q29: ConfigMap vs Secret -------------------------------------------
    "005a4025a052": ("inferred", "The ConfigMap half is stated. No premise describes what a Secret holds; 'passwords or OAuth tokens' appears nowhere. It follows from 'use a Secret rather than a ConfigMap' for confidential data, but is not stated.", False),

    # --- q30: taints and tolerations ----------------------------------------
    "fb7e1cea7e89": ("grounded", "Premise: 'The control plane, using the node controller, automatically creates taints with a NoSchedule effect for node conditions.' The two examples are not enumerated in the premise, but the assertion itself is stated.", False),
    "86b9e60e7af5": ("grounded", "VERBATIM: 'The scheduler checks taints, not node conditions, when it makes scheduling decisions.' Tagged uncertain only because two unrelated premises score contradiction 1.00 and the rule takes the max over all premises.", False),
    "3ff3ebdbe8af": ("grounded", "Premise: 'Tolerations allow Pods to be scheduled on nodes with matching taints', with the YAML that does it.", False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="overwrite grounding.jsonl, discarding hand edits")
    args = ap.parse_args()
    if OUT.exists() and not args.force:
        sys.exit(f"{OUT} already exists and is the source of truth.\n"
                 f"Edit it directly to correct a label. Pass --force only if "
                 f"you mean to regenerate it and lose any edits since.")
    if not POOL.exists():
        sys.exit(f"{POOL} missing — run: python eval/harvest_labels.py answers")

    pool = {}
    for line in POOL.open(encoding="utf-8"):
        r = json.loads(line)
        pool[r["sid"]] = r

    scorable = {sid for sid, r in pool.items() if not r["skipped_by_code"]}

    unknown = LABELS.keys() - pool.keys()
    if unknown:
        sys.exit(f"labelled sids not in pool: {sorted(unknown)}")
    mislabelled = LABELS.keys() - scorable
    if mislabelled:
        sys.exit(f"labels given for sentences the code skips: {sorted(mislabelled)}")
    unlabelled = scorable - LABELS.keys()
    if unlabelled:
        sys.exit(f"{len(unlabelled)} scorable sentences still unlabelled: "
                 f"{sorted(unlabelled)}")

    rows = []
    for sid, (label, note, miss) in LABELS.items():
        r = pool[sid]
        row = {
            "sid": sid,
            "label": label,
            "note": note,
            "qid": r["qid"],
            "question": r["question"],
            "sentence": r["sentence"],
            "claim": r["claim"],
            "cited": r["cited"],
            "premises": [{"source": p["source"], "section": p["section"],
                          "text": p["text"]} for p in r["premises"]],
        }
        if miss:
            row["retrieval_miss"] = True
        rows.append(row)

    rows.sort(key=lambda r: (r["qid"], r["sid"]))
    with OUT.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"{len(rows)} sentences -> {OUT}")
    for tag in ("grounded", "inferred", "uncertain"):
        n = sum(1 for r in rows if r["label"] == tag)
        print(f"  {tag:10s} {n:3d}")
    print(f"  retrieval_miss flagged: {sum(1 for r in rows if r.get('retrieval_miss'))}")
    print(f"  skipped by _is_claim (not labelled): {len(pool) - len(scorable)}")


if __name__ == "__main__":
    main()
