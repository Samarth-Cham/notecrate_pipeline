"""
Choose the Week 4 thresholds from labelled data instead of guessing them.

  python eval/calibrate.py conflicts     # CONTRADICTION_THRESHOLD
  python eval/calibrate.py grounding     # GROUNDED / INFERRED / CONTRADICTED
  python eval/calibrate.py both

Reads the hand-labelled sets in eval/labels/, re-scores them with the same
NLI model production uses, and sweeps each threshold over its range so the
constant can be read off a curve.

Both label files are self-contained: they carry the chunk and premise text
they were labelled against, so a sweep reproduces without the database. The
NLI scores are cached in _scores.json, keyed by a hash of (model, premise,
hypothesis) — editing a label never invalidates a score, and changing the
model or the text invalidates exactly the affected rows.

READ THIS BEFORE QUOTING A PRECISION NUMBER
-------------------------------------------
Genuine contradictions are rare in this corpus — it is curated Kubernetes
docs plus chat transcripts, which mostly agree with themselves. The labelled
positives are therefore padded with `perturbed` pairs: real chunks with one
fact minimally negated. Those are legitimate for measuring RECALL ("if a
contradiction existed, would this threshold catch it?") but they inflate the
base rate, so precision over the whole set is meaningless. This script
reports precision over corpus-origin pairs only, and prints recall on the
perturbed ones separately. Do not average them together.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import conflict as conflict_mod
from src import verify as verify_mod
from src.nli import MAX_CHARS, NLI_MODEL, score as nli_score

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "eval" / "labels"
CONFLICTS = LABELS / "conflicts.jsonl"
GROUNDING = LABELS / "grounding.jsonl"
CACHE = LABELS / "_scores.json"

GRID = [round(0.05 * i, 2) for i in range(1, 20)]   # 0.05 .. 0.95

# A contradiction probability can reach 1.0, so 0.95 does not switch the
# contradiction veto off — it still fires on everything that saturates. The
# sweep needs to be able to express "disable this rule" or it cannot tell you
# that the rule is the problem.
OFF = 1.01
CONTRA_GRID = GRID + [OFF]

# A disagreement panel that cries wolf is worse than one that stays quiet:
# a false positive also flips pipeline.py into CONFLICT_SYSTEM, which makes
# the model hedge on sources that agree fine. So the recommended threshold is
# the most sensitive one that still clears this precision bar.
MIN_PRECISION = 0.90


# --- scoring with a content-addressed cache ---------------------------------

def _key(premise: str, hypothesis: str) -> str:
    raw = f"{NLI_MODEL}\x00{premise[:MAX_CHARS]}\x00{hypothesis[:MAX_CHARS]}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def score_pairs(pairs: list[tuple[str, str]]) -> list[dict]:
    """NLI scores for `pairs`, computing only what the cache is missing."""
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}

    missing = [p for p in pairs if _key(*p) not in cache]
    # Dedupe: the same premise/hypothesis often appears under several labels.
    unique = list(dict.fromkeys(missing))
    if unique:
        print(f"  scoring {len(unique)} uncached pairs with {NLI_MODEL}...")
        for pair, result in zip(unique, nli_score(unique)):
            cache[_key(*pair)] = {k: round(v, 6) for k, v in result.items()}
        CACHE.write_text(json.dumps(cache, indent=0, sort_keys=True), encoding="utf-8")

    return [cache[_key(*p)] for p in pairs]


def read_labels(path: Path, expected: set[str]) -> list[dict]:
    if not path.exists():
        sys.exit(f"{path} does not exist — run eval/harvest_labels.py and label the pool.")
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    unlabelled = [r for r in rows if r.get("label") not in expected]
    if unlabelled:
        sys.exit(f"{path}: {len(unlabelled)} rows have a label outside {sorted(expected)} "
                 f"(first: {unlabelled[0].get('pid') or unlabelled[0].get('sid')})")
    return rows


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


# --- conflict threshold -----------------------------------------------------

def calibrate_conflicts() -> None:
    rows = read_labels(CONFLICTS, {"conflict", "agree", "unrelated"})

    # Score both directions, exactly as conflict.py does, so the sweep can
    # also re-test the min-vs-max decision on labelled data.
    forward = score_pairs([(r["a_text"], r["b_text"]) for r in rows])
    reverse = score_pairs([(r["b_text"], r["a_text"]) for r in rows])

    for r, f, b in zip(rows, forward, reverse):
        r["_min"] = min(f["contradiction"], b["contradiction"])
        r["_max"] = max(f["contradiction"], b["contradiction"])

    positives = [r for r in rows if r["label"] == "conflict"]
    negatives = [r for r in rows if r["label"] != "conflict"]
    corpus = [r for r in rows if r.get("origin", "corpus") == "corpus"]
    perturbed = [r for r in rows if r.get("origin") == "perturbed"]

    print(f"\n{'=' * 78}\nCONFLICT THRESHOLD\n{'=' * 78}")
    print(f"{len(rows)} labelled pairs: {len(positives)} conflict, "
          f"{sum(1 for r in negatives if r['label'] == 'agree')} agree, "
          f"{sum(1 for r in negatives if r['label'] == 'unrelated')} unrelated")
    print(f"origin: {len(corpus)} corpus, {len(perturbed)} perturbed "
          f"(perturbed excluded from precision)")

    for direction in ("_min", "_max"):
        name = "min(both directions)  [current code]" if direction == "_min" \
            else "max(either direction)"
        print(f"\n--- {name} ---")
        print(f"{'':>7} {'--- corpus-origin pairs ---':^24} "
              f"{'--- all pairs ---':^22}")
        print(f"{'thresh':>7} {'prec':>7} {'recall':>7} {'F1':>8} "
              f"{'recall':>7} {'TP':>4} {'FP':>4} {'FN':>4}")

        best = None
        for t in GRID:
            # Precision, recall and F1 all over corpus-origin pairs only, so
            # the three describe one population and F1 means something. Mixing
            # in perturbed positives would inflate recall against a base rate
            # that was invented — see the module docstring.
            c_tp = sum(1 for r in corpus if r["label"] == "conflict" and r[direction] >= t)
            c_fp = sum(1 for r in corpus if r["label"] != "conflict" and r[direction] >= t)
            c_fn = sum(1 for r in corpus if r["label"] == "conflict" and r[direction] < t)
            precision, c_recall, f1 = prf(c_tp, c_fp, c_fn)

            # Separately: recall over every positive, perturbed included. This
            # answers "would a contradiction be caught if one existed", which
            # the corpus alone has too few positives to answer.
            tp = sum(1 for r in positives if r[direction] >= t)
            fn = len(positives) - tp
            fp = sum(1 for r in negatives if r[direction] >= t)
            _, recall_all, _ = prf(tp, fp, fn)

            flag = ""
            if precision >= MIN_PRECISION and c_tp and best is None:
                best, flag = t, "  <-- lowest threshold clearing precision bar"
            print(f"{t:7.2f} {precision:7.2f} {c_recall:7.2f} {f1:8.2f} "
                  f"{recall_all:7.2f} {tp:4d} {fp:4d} {fn:4d}{flag}")

        if direction == "_min":
            print(f"\n  precision/recall/F1 over the {len(corpus)} corpus-origin "
                  f"pairs; the right-hand columns cover all {len(rows)}, "
                  f"perturbed positives included")
            current = conflict_mod.CONTRADICTION_THRESHOLD
            print(f"  current CONTRADICTION_THRESHOLD = {current}")
            if best is not None:
                print(f"  suggested = {best}" +
                      ("" if best == current else f"  (change from {current})"))
            else:
                print(f"  no threshold reaches precision {MIN_PRECISION} with any "
                      f"true positive — the detector cannot be made trustworthy "
                      f"by thresholding alone on this data")

    if perturbed:
        print("\n--- recall on perturbed positives only (min direction) ---")
        pos_p = [r for r in perturbed if r["label"] == "conflict"]
        for t in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
            hit = sum(1 for r in pos_p if r["_min"] >= t)
            print(f"{t:7.2f} {hit:3d}/{len(pos_p)}")

    _misses(rows, "_min", conflict_mod.CONTRADICTION_THRESHOLD)


def _misses(rows: list[dict], direction: str, threshold: float) -> None:
    """The rows the current constant gets wrong — the useful output."""
    fp = [r for r in rows if r["label"] != "conflict" and r[direction] >= threshold]
    fn = [r for r in rows if r["label"] == "conflict" and r[direction] < threshold]

    if fp:
        print(f"\nFALSE POSITIVES at {threshold} ({len(fp)}):")
        for r in fp:
            print(f"  [{r[direction]:.2f}] {r['label']:9s} {r['a_source']} vs {r['b_source']}")
            if r.get("note"):
                print(f"           {r['note']}")
    if fn:
        print(f"\nMISSED CONFLICTS at {threshold} ({len(fn)}):")
        for r in fn:
            print(f"  [{r[direction]:.2f}] {r.get('origin', 'corpus'):9s} "
                  f"{r['a_source']} vs {r['b_source']}")
            if r.get("note"):
                print(f"           {r['note']}")


# --- grounding thresholds ---------------------------------------------------

TAGS = ("grounded", "inferred", "uncertain")

# "[2] states that ...", "according to [3], ...", "* [4] claims ..." — the
# attribution shapes CONFLICT_SYSTEM asks the generator for.
CITED_CLAIM_RE = re.compile(
    r"\[\d+(?:\s*,\s*\d+)*\]\s*(?:\w+\s+)?"
    r"(?:states?|says?|notes?|describes?|mentions?|indicates?|explains?|"
    r"claims?|shows?|suggests?)\b"
    r"|according to\s+\[\d+",
    re.IGNORECASE,
)


def _tag(entail: float, contra: float, grounded: float,
         inferred: float, contradicted: float) -> str:
    """verify.py's decision rule, parameterised. Keep in sync."""
    if contra >= contradicted and contra > entail:
        return "uncertain"
    if entail >= grounded:
        return "grounded"
    if entail >= inferred:
        return "inferred"
    return "uncertain"


def calibrate_grounding() -> None:
    rows = read_labels(GROUNDING, set(TAGS))

    # Re-derive the hypothesis with the LIVE strip_citations rather than using
    # the `claim` stored at labelling time. That field is derived data sitting
    # in a hand-labelled file: the moment someone improves the attribution
    # regexes in verify.py, a stored copy calibrates the old behaviour. The
    # human label is about the sentence and its premises, so it stays valid
    # across such a change — only the text handed to the model should move.
    for r in rows:
        r["_hypothesis"] = verify_mod.strip_citations(r["sentence"])

    drifted = [r for r in rows if r["_hypothesis"] != r.get("claim")]
    if drifted:
        print(f"note: strip_citations has changed since labelling — "
              f"{len(drifted)} of {len(rows)} hypotheses differ from the stored "
              f"`claim`. Scoring the current output, which is the point.")

    pairs, spans = [], []
    for r in rows:
        start = len(pairs)
        pairs.extend((p["text"], r["_hypothesis"]) for p in r["premises"])
        spans.append((start, len(pairs)))

    scored = score_pairs(pairs)
    for r, (start, end) in zip(rows, spans):
        window = scored[start:end]
        if not window:
            r["_entail"] = r["_contra_max"] = r["_contra_best"] = 0.0
            continue
        # Mirror verify.py: entailment of the best-entailing premise, but
        # contradiction is the max over ALL premises.
        best = max(range(len(window)), key=lambda i: window[i]["entailment"])
        r["_entail"] = window[best]["entailment"]
        r["_contra_max"] = max(s["contradiction"] for s in window)
        # The alternative worth measuring: judge contradiction on the premise
        # the sentence is actually being grounded against, so an unrelated
        # chunk that retrieval happened to return cannot veto the tag.
        r["_contra_best"] = window[best]["contradiction"]

    print(f"\n{'=' * 78}\nGROUNDING THRESHOLDS\n{'=' * 78}")
    counts = {t: sum(1 for r in rows if r["label"] == t) for t in TAGS}
    print(f"{len(rows)} labelled sentences: " +
          ", ".join(f"{n} {t}" for t, n in counts.items()))

    current = (verify_mod.GROUNDED_THRESHOLD, verify_mod.INFERRED_THRESHOLD,
               verify_mod.CONTRADICTED_THRESHOLD)
    print(f"current (grounded, inferred, contradicted) = {current}")
    _report_grounding(rows, *current, header="CURRENT THRESHOLDS")

    # Two candidate rules, not just two thresholds. `max` is what verify.py
    # does today; `best` scores contradiction on the premise the sentence is
    # actually grounded against. If the second wins by a wide margin, the fix
    # is in the rule, and no amount of threshold tuning substitutes for it.
    results = {}
    for contra_key in ("_contra_max", "_contra_best"):
        best = None
        for grounded in GRID:
            for inferred in GRID:
                if inferred >= grounded:
                    continue
                for contradicted in CONTRA_GRID:
                    acc, macro = _grounding_scores(rows, grounded, inferred,
                                                   contradicted, contra_key)
                    # Macro-F1, not accuracy: the set is imbalanced and
                    # accuracy alone rewards never predicting the rare tag.
                    if best is None or macro > best[0]:
                        best = (macro, acc, grounded, inferred, contradicted)
        results[contra_key] = best

    print(f"\n{'-' * 78}\nBEST ACHIEVABLE, BY CONTRADICTION RULE")
    print(f"{'rule':>34} {'grounded':>9} {'inferred':>9} {'contradicted':>13} "
          f"{'acc':>6} {'macroF1':>8}")
    for key, (macro, acc, g, i, c) in results.items():
        name = ("max over all premises (current)" if key == "_contra_max"
                else "best-entailing premise only")
        shown = "off" if c >= OFF else f"{c}"
        print(f"{name:>34} {g:9.2f} {i:9.2f} {shown:>13} {acc:6.3f} {macro:8.3f}")

    winner = max(results, key=lambda k: results[k][0])
    macro, acc, grounded, inferred, contradicted = results[winner]
    rule = ("max over all premises" if winner == "_contra_max"
            else "contradiction from the best-entailing premise")
    print(f"\nBEST OVERALL: {rule}, grounded={grounded} inferred={inferred} "
          f"contradicted={'off' if contradicted >= OFF else contradicted}")
    _report_grounding(rows, grounded, inferred, contradicted,
                      header="SUGGESTED", contra_key=winner)

    print("\nSensitivity of GROUNDED_THRESHOLD (others held at suggested):")
    print(f"{'thresh':>7} {'acc':>6} {'macroF1':>8}   grounded P/R")
    for t in GRID:
        if t <= inferred:
            continue
        acc_t, macro_t = _grounding_scores(rows, t, inferred, contradicted, winner)
        tp = sum(1 for r in rows if r["label"] == "grounded"
                 and _tag(r["_entail"], r[winner], t, inferred, contradicted) == "grounded")
        fp = sum(1 for r in rows if r["label"] != "grounded"
                 and _tag(r["_entail"], r[winner], t, inferred, contradicted) == "grounded")
        fn = counts["grounded"] - tp
        p, rec, _ = prf(tp, fp, fn)
        mark = "  <-- suggested" if t == grounded else ""
        print(f"{t:7.2f} {acc_t:6.2f} {macro_t:8.3f}   {p:.2f}/{rec:.2f}{mark}")

    _attribution_breakdown(rows, grounded, inferred, contradicted, winner)


def _attribution_breakdown(rows, grounded, inferred, contradicted, contra_key):
    """Split accuracy by whether the sentence attributes a claim to a source.

    The disagreement panel makes the generator write "[2] states that X"
    constantly. Those sentences assert something about the CORPUS, which is a
    proposition no chunk entails, so they are systematically hard to ground.
    Worth seeing separately: if accuracy collapses on them, the fix is
    upstream in prompting, not in these thresholds.
    """
    attributed = [r for r in rows if CITED_CLAIM_RE.search(r["sentence"])]
    plain = [r for r in rows if not CITED_CLAIM_RE.search(r["sentence"])]

    print("\nAccuracy by sentence shape:")
    for name, subset in (("attributes a claim to a source", attributed),
                         ("plain assertion", plain)):
        if not subset:
            continue
        acc, macro = _grounding_scores(subset, grounded, inferred,
                                       contradicted, contra_key)
        n_grounded = sum(1 for r in subset if r["label"] == "grounded")
        print(f"  {name:32s} n={len(subset):3d}  acc={acc:.3f}  "
              f"macroF1={macro:.3f}  ({n_grounded} labelled grounded)")


def _grounding_scores(rows: list[dict], grounded: float, inferred: float,
                      contradicted: float, contra_key: str) -> tuple[float, float]:
    predicted = [_tag(r["_entail"], r[contra_key], grounded, inferred, contradicted)
                 for r in rows]
    correct = sum(1 for r, p in zip(rows, predicted) if r["label"] == p)
    f1s = []
    for tag in TAGS:
        tp = sum(1 for r, p in zip(rows, predicted) if r["label"] == tag and p == tag)
        fp = sum(1 for r, p in zip(rows, predicted) if r["label"] != tag and p == tag)
        fn = sum(1 for r, p in zip(rows, predicted) if r["label"] == tag and p != tag)
        f1s.append(prf(tp, fp, fn)[2])
    return correct / len(rows), sum(f1s) / len(f1s)


def _report_grounding(rows: list[dict], grounded: float, inferred: float,
                      contradicted: float, *, header: str,
                      contra_key: str = "_contra_max") -> None:
    predicted = [_tag(r["_entail"], r[contra_key], grounded, inferred, contradicted)
                 for r in rows]
    acc, macro = _grounding_scores(rows, grounded, inferred, contradicted, contra_key)

    shown = "off" if contradicted >= OFF else contradicted
    print(f"\n--- {header}: ({grounded}, {inferred}, {shown}) ---")
    print(f"tag accuracy {acc:.3f}   macro-F1 {macro:.3f}")

    print(f"\n{'':>12}" + "".join(f"{('pred ' + t)[:10]:>11}" for t in TAGS))
    for tag in TAGS:
        cells = "".join(
            f"{sum(1 for r, p in zip(rows, predicted) if r['label'] == tag and p == t):>11d}"
            for t in TAGS)
        print(f"{('true ' + tag):>12}{cells}")

    # A grid search on 54 rows will happily "win" by collapsing the problem —
    # dropping a tag entirely, or setting GROUNDED so low that almost
    # everything clears it. Those score well and mean nothing, so say it.
    predicted_counts = {t: predicted.count(t) for t in TAGS}
    dominant = max(predicted_counts, key=predicted_counts.get)
    warnings = []

    # A tag with zero F1 is dead whether or not the rule ever emits it: the
    # three-way distinction the UI renders has quietly become two-way.
    dead = []
    for tag in TAGS:
        tp = sum(1 for r, p in zip(rows, predicted) if r["label"] == tag and p == tag)
        if tp == 0 and any(r["label"] == tag for r in rows):
            dead.append(f"{tag} (predicted {predicted_counts[tag]}x, never correctly)")
    if dead:
        warnings.append("gets no " + " or ".join(dead))
    if predicted_counts[dominant] / len(rows) > 0.75:
        warnings.append(f"predicts {dominant} for "
                        f"{predicted_counts[dominant]}/{len(rows)} sentences")
    if warnings:
        print(f"  DEGENERATE: this configuration {'; '.join(warnings)}. "
              f"It scores well by collapsing the task, not by solving it.")

    wrong = [(r, p) for r, p in zip(rows, predicted) if r["label"] != p]
    if wrong:
        print(f"\nMISCLASSIFIED ({len(wrong)}):")
        for r, p in wrong:
            print(f"  human={r['label']:9s} system={p:9s} "
                  f"e={r['_entail']:.2f} c={r[contra_key]:.2f}  {r['sentence'][:60]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="both",
                    choices=["conflicts", "grounding", "both"])
    args = ap.parse_args()

    if args.what in ("conflicts", "both"):
        calibrate_conflicts()
    if args.what in ("grounding", "both"):
        calibrate_grounding()


if __name__ == "__main__":
    main()
