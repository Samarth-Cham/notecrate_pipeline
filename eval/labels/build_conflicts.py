"""Regenerate eval/labels/conflicts.jsonl from chunk ids + the labels below.

  python eval/labels/build_conflicts.py --force

This is how the file was first built: it pulls chunk text from the database so
the label file ends up self-contained, without anyone hand-copying 1KB of text
per row. Perturbations are expressed as explicit (find, replace) pairs, so the
edit is auditable — one fact changed, everything else byte-identical.

conflicts.jsonl is the source of truth once it exists. Correcting a label
means editing that file, and this script refuses to overwrite it without
--force so a rebuild cannot silently discard those corrections. Needs the
corpus indexed in Postgres; calibrate.py does not.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

OUT = ROOT / "eval" / "labels" / "conflicts.jsonl"

# --- perturbations: (id, name, [(find, replace)], description) --------------
PERTURBATIONS = {
    "configmap_ns": (
        2186,
        [("The Pod and the ConfigMap must be in the\nsame namespace.",
          "A Pod may reference a ConfigMap from any namespace in the cluster; the\ntwo do not need to share a namespace.")],
        "inverts the same-namespace requirement for Pod -> ConfigMap references",
    ),
    "job_restartpolicy": (
        2470,
        [("Only a [`RestartPolicy`](/docs/concepts/workloads/pods/pod-lifecycle/#restart-policy)\nequal to `Never` or `OnFailure` is allowed.",
          "Only a [`RestartPolicy`](/docs/concepts/workloads/pods/pod-lifecycle/#restart-policy)\nequal to `Always` is allowed; `Never` and `OnFailure` are rejected by validation.")],
        "swaps the allowed Job restartPolicy values to their opposite",
    ),
    "taint_repel": (
        2937,
        [("_Taints_ are the opposite -- they allow a node to repel a set of pods.",
          "_Taints_ work the same way -- they allow a node to attract a set of pods.")],
        "inverts taint semantics from repel to attract",
    ),
    "init_completion": (
        2423,
        [("* Init containers always run to completion.\n* Each init container must complete successfully before the next one starts.",
          "* Init containers are started in parallel with the app containers.\n* An init container need not complete before the next one, or the app containers, start.")],
        "inverts the ordering and run-to-completion guarantee",
    ),
    "statefulset_stable": (
        2906,
        [("In the above, stable is synonymous with persistence across Pod (re)scheduling.",
          "In the above, identifiers are reassigned at random on every Pod (re)scheduling and do not persist.")],
        "inverts the stability guarantee that defines a StatefulSet",
    ),
    "hpa_max": (
        2341,
        [("The HorizontalPodAutoscaler takes the maximum scale\nrecommended for each metric",
          "The HorizontalPodAutoscaler takes the minimum scale\nrecommended for each metric")],
        "swaps max for min in multi-metric scale selection",
    ),
    "namespace_unique": (
        2585,
        [("Names of resources need to be unique within a namespace, but not across namespaces.",
          "Names of resources must be unique across the entire cluster, regardless of namespace.")],
        "inverts the scope of the name-uniqueness rule",
    ),
}

# --- hand labels ------------------------------------------------------------
# (a, b, label, origin, note)   a/b are chunk ids, or perturbation keys.
PAIRS = [
    # --- real conflicts already in the corpus --------------------------------
    (861, 862, "conflict", "corpus",
     "861 states India's gold import duty is ~6%; 862 states the May 2026 hike "
     "took it to 15%. Same fact, same conversation, superseded value never "
     "retracted. This is the pattern the disagreement panel exists for."),
    (861, 869, "conflict", "corpus",
     "Same 6%-vs-15% conflict as (861,862); 869 makes the supersession explicit "
     "('reversed a liberalisation from July 2024 that had cut the duty from 15% "
     "down to 6%'). Correlated with (861,862) — one underlying fact, not two "
     "independent positives."),
    (861, 2641, "conflict", "corpus",
     "Cross-document version of the same conflict: chat says ~6% duty, the "
     "generated PDF report says 15%. Different source_type, so retrieval can "
     "surface them together for a single question."),

    # --- perturbed conflicts -------------------------------------------------
    (2186, "configmap_ns", "conflict", "perturbed", "same-namespace requirement, inverted"),
    (2470, "job_restartpolicy", "conflict", "perturbed", "allowed Job restartPolicy, inverted"),
    (2937, "taint_repel", "conflict", "perturbed", "taint repel/attract, inverted"),
    (2423, "init_completion", "conflict", "perturbed", "init container ordering guarantee, inverted"),
    (2906, "statefulset_stable", "conflict", "perturbed", "StatefulSet stability guarantee, inverted"),
    (2341, "hpa_max", "conflict", "perturbed", "HPA multi-metric max/min, inverted"),
    (2585, "namespace_unique", "conflict", "perturbed", "name uniqueness scope, inverted"),

    # --- hard negatives: same topic, reconcilable ----------------------------
    # NOTE: (861,862) and (861,869) are also present in the retrieved pool and
    # are deduped against POOL_LABELS below — same content, same pid.
    (2423, 2899, "agree", "corpus",
     "HARDEST NEGATIVE IN THE SET. init-containers.md: 'Init containers always "
     "run to completion'. sidecar-containers.md: an init container with "
     "restartPolicy Always 'remains running during the entire life of the Pod'. "
     "Propositionally a contradiction; actually the documented special case. "
     "Labelled agree per the rubric — no reader is misled, and flagging it "
     "would fire CONFLICT_SYSTEM on two docs that belong together."),
    (862, 869, "agree", "corpus",
     "Both describe the May 2026 duty hike to 15%. 869 adds the 2024 history. "
     "Complementary detail, not disagreement."),
    (862, 878, "agree", "corpus",
     "Both give the 15% figure and the 4-6% actual price move. Restatement."),
    (2641, 2642, "agree", "corpus",
     "Consecutive chunks of the same PDF section on duty-reversal risk."),
    (2186, 2188, "agree", "corpus",
     "Two chunks of configmap.md on how Pods consume ConfigMaps."),
    (2906, 2585, "agree", "corpus",
     "StatefulSet stability vs namespace scoping — both k8s concepts, different "
     "subjects, but enough shared vocabulary to be a plausible NLI trap."),
    (2340, 2341, "agree", "corpus",
     "HPA custom metrics vs multiple metrics: adjacent doc sections that share "
     "an opening sentence verbatim."),
    (2470, 2423, "agree", "corpus",
     "Job restartPolicy constraint vs init container restart behaviour. Both "
     "about restartPolicy, neither contradicts the other."),
]


# --- labels for pairs the retriever actually produced -----------------------
# Every pool pair scoring >= 0.10 (all 20 of them), plus a seeded sample of the
# near-zero band and the real conflicts that landed there. This is the honest
# negative distribution: what the detector sees in production.
POOL_LABELS = {
    # -- everything the detector ranks highest -------------------------------
    "700678bb510a": ("agree", "Two chunks of one heapsort walkthrough: a build-max-heap trace and the two-phase summary. Complementary."),
    "aba1e5f8201b": ("unrelated", "Merge-sort midpoint arithmetic vs a linear-algebra homework sheet. Retrieved together only because the least-squares question matched nothing well."),
    "370fd2448e65": ("agree", "Two explanations of building a max heap, from different conversations. Both correct."),
    "10c170bcf47d": ("unrelated", "QuickSort partition trace vs an FSM truth table. Retrieved for 'best pizza recipe' — a negative question, so both chunks are noise."),
    "65f19d12151d": ("unrelated", "Heapify trace vs a quicksort partition result table. Same conversation, different algorithms."),
    "a2553370f540": ("agree", "sidecar-containers.md vs init-containers.md on how the two differ. They agree — each describes its own kind."),
    "948538c3358d": ("unrelated", "Secret files on node disk vs persistent-volume plugin architecture. Both Windows docs, different subjects."),
    "5d862af453ef": ("agree", "Heapify trace vs buildHeap complexity summary. Same conversation, consistent."),
    "41657edcc044": ("agree", "Two QuickSort multiple-choice questions over different arrays. Different instances, no shared claim to contradict."),
    "4df4bbcf3584": ("agree", "sidecar-containers.md intro vs its own 'Differences from init containers' section."),
    "b5fe5a462267": ("unrelated", "Sidecar containers vs NUMA topology manager scopes."),
    "281dd410ce6a": ("unrelated", "Permutation pseudocode vs a frontend filtering bug. Retrieved for 'how do skip lists work', which the corpus does not cover."),
    "2495e9f20938": ("agree", "The heap-sort complexity question and its answer, in the same conversation."),
    "13023e44a0a1": ("unrelated", "Validating one specific red-black tree vs classifying a test as clear-box. Shared vocabulary, different subjects."),
    "217e4c19560b": ("agree", "'The repo does not exist yet, create it' vs 'generate a PAT'. Consecutive steps of one setup, not a disagreement."),
    "fa4cd8fa9a4c": ("agree", "Two options for creating a GitHub repo offered side by side in one answer."),
    "ef334b986b5c": ("agree", "The canonical near-miss: init containers run to completion and do not support probes; sidecars run concurrently and do. Each statement is scoped to its own container kind."),
    "3c4bec63a657": ("agree", "Two consecutive paragraphs of the same proof that bottom-up heap building is linear."),
    "b9b40fc9c063": ("agree", "Creating the repo via the GitHub API vs generating the token that call needs."),
    "6311a89b0690": ("unrelated", "Git branch workflow vs JUnit compilation flags."),

    # -- real conflicts that the detector scored near zero -------------------
    "2e0860893e0a": ("conflict", "REAL CONFLICT, MISSED. The stale '~6%' duty chunk retrieved alongside the chunk announcing the May 2026 hike to 15%, for the question that asks exactly this. Scored 0.002."),
    "2443bc89d61e": ("conflict", "REAL CONFLICT, MISSED. Same 6%-vs-15% disagreement, paired with the policy-paragraph chunk. Scored 0.001."),
    "8e5dfbee2062": ("conflict", "REAL CONFLICT, MISSED. Same disagreement, paired with the chunk that states the 6%->15% rise explicitly. Scored 0.011."),
    "9878b6915869": ("unrelated", "The stale-duty chunk paired with an illustrative P/L table that names no duty rate. No shared claim."),
    "9f24b6469f66": ("conflict", "REAL CONFLICT, MISSED (0.0004). One chunk recommends 'Chunk size: 900 tokens, Overlap: 150'; the other says 'chunk size (600, settled for now)'. Retrieved together for the question that asks which was chosen — and the generator asserted 900/150 as the decision without flagging the disagreement."),

    # -- seeded sample of the near-zero band ---------------------------------
    "d1292fdd1169": ("agree", "Two QuickSort partition examples."),
    "c0559ea3292c": ("agree", "Two parts of one clear-box vs I/O test explanation."),
    "f4c0111f83d4": ("agree", "A summary of download methods and the scp command it recommends."),
    "02799d216c57": ("agree", "QuickSort implementation vs a QuickSort quiz question."),
    "276512f1e985": ("agree", "Randomised red-black-tree testing vs validating a specific tree."),
    "97782b83fdf1": ("unrelated", "Toleration YAML vs node resource capacity."),
    "8bdbb9d60966": ("unrelated", "Init containers overview vs NUMA resource managers."),
    "bb0e1ddda10b": ("agree", "Hash table definition vs double hashing — same document, complementary."),
    "b63b467720fa": ("unrelated", "Merge-sort midpoint vs a maths homework sheet."),
    "8254cc52f73b": ("agree", "Two steps of one red-black-tree validity walkthrough."),
}

POOL = ROOT / "eval" / "labels" / "_pool_conflicts.jsonl"


def pool_rows():
    if not POOL.exists():
        sys.exit(f"{POOL} missing — run: python eval/harvest_labels.py conflicts")
    by_pid = {}
    for line in POOL.open(encoding="utf-8"):
        r = json.loads(line)
        by_pid[r["pid"]] = r

    missing = POOL_LABELS.keys() - by_pid.keys()
    if missing:
        sys.exit(f"labelled pids not in pool (was it re-harvested?): {sorted(missing)}")

    out = []
    for pid, (label, note) in POOL_LABELS.items():
        r = by_pid[pid]
        out.append({
            "pid": pid, "label": label, "origin": "corpus", "note": note,
            "a_source": r["a_source"], "a_section": r["a_section"], "a_text": r["a_text"],
            "b_source": r["b_source"], "b_section": r["b_section"], "b_text": r["b_text"],
            "retrieved_for": r["queries"][0],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="overwrite conflicts.jsonl, discarding hand edits")
    args = ap.parse_args()
    if OUT.exists() and not args.force:
        sys.exit(f"{OUT} already exists and is the source of truth.\n"
                 f"Edit it directly to correct a label. Pass --force only if "
                 f"you mean to regenerate it from this script and lose any "
                 f"edits made since.")

    ids = {x for pair in PAIRS for x in pair[:2] if isinstance(x, int)}
    ids |= {PERTURBATIONS[k][0] for pair in PAIRS for k in pair[:2]
            if isinstance(k, str)}

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        rows = conn.execute(
            "SELECT id, source, section, text FROM chunks WHERE id = ANY(%s)",
            (list(ids),)).fetchall()
    chunks = {r[0]: {"source": r[1], "section": r[2], "text": r[3]} for r in rows}

    missing = ids - chunks.keys()
    if missing:
        sys.exit(f"chunk ids not in DB: {sorted(missing)}")

    def resolve(ref):
        """-> (source, section, text, perturbed_from, perturbation)"""
        if isinstance(ref, int):
            c = chunks[ref]
            return c["source"], c["section"], c["text"], None, None
        src_id, edits, description = PERTURBATIONS[ref]
        c = chunks[src_id]
        text = c["text"]
        for find, replace in edits:
            # Match across the chunk's own line wrapping: the docs are hard
            # wrapped and where the break falls is not worth encoding here.
            pattern = r"\s+".join(re.escape(w) for w in find.split())
            text, n = re.subn(pattern, lambda _: replace, text, count=1)
            if n != 1:
                sys.exit(f"perturbation {ref!r}: find-string absent from chunk "
                         f"{src_id}. Chunk text:\n{c['text']!r}")
        return c["source"], c["section"], text, c["source"], description

    out = []
    for a, b, label, origin, note in PAIRS:
        a_source, a_section, a_text, a_from, a_pert = resolve(a)
        b_source, b_section, b_text, b_from, b_pert = resolve(b)

        key = "\x00".join(sorted([a_text.strip(), b_text.strip()]))
        row = {
            "pid": hashlib.sha1(key.encode()).hexdigest()[:12],
            "label": label,
            "origin": origin,
            "note": note,
            "a_source": a_source, "a_section": a_section, "a_text": a_text,
            "b_source": b_source, "b_section": b_section, "b_text": b_text,
        }
        if origin == "perturbed":
            row["perturbed_from"] = b_from or a_from
            row["perturbation"] = b_pert or a_pert
        out.append(row)

    # Pool rows win on collision: they carry `retrieved_for`, and a pair that
    # the retriever actually produced is the more informative record of it.
    manual = {row["pid"]: row for row in out}
    merged = {}
    for row in pool_rows():
        merged[row["pid"]] = row
    for pid, row in manual.items():
        if pid in merged:
            if merged[pid]["label"] != row["label"]:
                sys.exit(f"pid {pid} labelled {row['label']} by hand but "
                         f"{merged[pid]['label']} in the pool — resolve before building")
            continue
        merged[pid] = row
    out = list(merged.values())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for row in out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"{len(out)} pairs -> {OUT}")
    for label in ("conflict", "agree", "unrelated"):
        n = sum(1 for r in out if r["label"] == label)
        c = sum(1 for r in out if r["label"] == label and r["origin"] == "corpus")
        print(f"  {label:9s} {n:3d}  ({c} corpus, {n - c} perturbed)")


if __name__ == "__main__":
    main()
