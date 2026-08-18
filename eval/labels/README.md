# Hand-labelled validation set

The plan (Section 9, Risks) puts it plainly:

> Conflict/grounding detection too noisy -> Hand-label a small validation set
> to calibrate thresholds before building UI around it.

Both Week 4 detectors shipped with guessed constants. This directory holds the
labels that turn them into measured ones.

```
answers.jsonl        frozen generated answers — the grounding substrate
conflicts.jsonl      chunk pairs      -> conflict | agree | unrelated
grounding.jsonl      answer sentences -> grounded | inferred | uncertain
build_*.py           regenerate the two label files (--force; see below)
_scores.json         cached NLI scores (derived; safe to delete, slow to rebuild)
_pool_*.jsonl        unlabelled candidates from harvest_labels.py (gitignored)
```

```bash
python eval/harvest_labels.py conflicts    # rebuild the candidate pools
python eval/harvest_labels.py answers
python eval/calibrate.py                   # sweep both sets, print curves
```

The `.jsonl` files are the source of truth. `build_conflicts.py` and
`build_grounding.py` are how they were first assembled — they pull chunk text
from Postgres so each row is self-contained — and both refuse to overwrite an
existing file without `--force`, so a rebuild cannot quietly discard a
correction someone made by hand.

## What the sweep found

Short version: **neither detector is merely miscalibrated.** The conflict
threshold cannot be fixed by choosing a different number, and the grounding
thresholds were already close to their achievable optimum.

**Conflicts — precision 0.00 at every threshold from 0.05 to 0.95.**
Not one corpus-origin conflict outranks a single negative. At the shipped
0.60, 14 of 51 pairs are flagged and all 14 are wrong: unrelated topic pairs
(a QuickSort trace against an FSM truth table) and same-topic pairs that
plainly agree (`sidecar-containers.md` against `init-containers.md`).
Meanwhile all five real conflicts in the set score below 0.02 — including the
two that retrieval genuinely surfaced together for the question that asks
about them. The `min`-of-both-directions rule in `conflict.py` does not fix
the unrelated-text failure its docstring says it fixed; measured over 298
retrieved pairs rather than 60, the failure is fully present.

**Grounding — tag accuracy 0.537 at the shipped thresholds; 0.630 at the
best setting found, and that setting is degenerate.**

The first thing the label set produced was not a threshold at all. Two
attribution-stripping bugs showed up in the `note` fields — `strip_citations`
only removed `[2] claims that X` when the word "that" was present, and never
handled the unmarked `another source suggests that X` — leaving nine
hypotheses mangled or wrapped in a claim about the corpus. Fixing those
regexes moved the numbers more than any threshold does:

| | before the regex fix | after |
|---|---|---|
| accuracy at shipped thresholds | 0.500 | **0.537** |
| best accuracy the sweep can reach | 0.519 | **0.630** |
| AUC, `grounded` vs `uncertain` | 0.598 | **0.724** |
| `grounded` sentences scoring < 0.05 | 11 of 35 | **6 of 35** |

`GROUNDED_THRESHOLD = 0.55` remains a reasonable cut. What does not work is
the middle tag. Entailment separates the classes like this:

```
grounded vs uncertain   AUC 0.724
grounded vs inferred    AUC 0.637   <-- the boundary INFERRED_THRESHOLD draws
inferred vs uncertain   AUC 0.548
```

Median entailment is 0.978 for `grounded` and 0.956 for `inferred`. The two
distributions sit on top of each other, so no cut between them can work, and
`calibrate.py` flags every configuration it finds — including the shipped one
— as **DEGENERATE: gets no inferred, never correctly**. The three-way tag the
UI renders is really a two-way decision with a third label that never lands.
Splitting `grounded` from `inferred` needs a second signal, not a better cut
point on this one.

Two further causes of wrong tags remain, both visible in the `note` fields
and neither fixable by tuning:

1. *Truncation.* `MAX_CHARS = 900` cuts the supporting sentence out of long
   chunks before the model sees it. One labelled row is grounded by text
   sitting near character 1,000 of its premise.
2. *The contradiction veto takes a max over all premises*, so a single
   unrelated retrieved chunk vetoes a correct tag. "The scheduler checks
   taints, not node conditions, when it makes scheduling decisions" is quoted
   verbatim from the corpus and comes out `uncertain` because two unrelated
   premises score contradiction 1.00. `calibrate.py` measures the alternative
   (score contradiction on the best-entailing premise only); it wins, but by
   little enough that 54 sentences cannot justify the change on its own.

And the two detectors fail *together*: `CONFLICT_SYSTEM` is what makes the
generator write attribution-shaped sentences, and it is switched on by
conflicts that are all false positives. 11 of 54 sentences have that shape
and score worst as a group (accuracy 0.455 against 0.674 for plain
assertions). Fixing conflict detection removes the sentences that grounding
handles worst.

## Who labelled this

**Claude (claude-opus-5) produced every label in this directory**, by reading
the chunk and premise text stored alongside each row. They are not human
labels. They were made carefully and against a written rubric, but a model
grading a model's output shares blind spots with it, and the whole point of
the exercise is an independent check.

Treat this as a first pass that makes the thresholds *measurable* rather than
guessed. Before anything user-facing is built on these numbers, spot-check
them — the `note` field on each row explains the reasoning, so disagreements
are quick to find. Rows worth checking first are the ones `calibrate.py`
prints under `MISCLASSIFIED` and `FALSE POSITIVES`: those are exactly where a
wrong label would move a threshold.

## Why answers.jsonl is frozen

Generation runs at `temperature=0.2`. Non-zero means re-running the pipeline
produces different sentences, and a grounding label pinned to a sentence that
no longer exists is worthless. So the answers were generated once and written
verbatim, and every downstream step reads that file.

`harvest_labels.py answers` reuses the frozen file by default and only
regenerates under `--regenerate`, which invalidates the labels. Re-running it
is a decision, not a side effect.

## Rubric: conflicts.jsonl

Label the *pair*, from the two texts stored in the row.

| label | meaning |
|---|---|
| `conflict` | The two chunks make claims that cannot both be true of the same subject. A reader who believed one would be misinformed by the other. Showing both is *necessary*. |
| `agree` | Same subject, compatible claims. Includes complementary detail, different levels of abstraction, and one text being a special case of the other. |
| `unrelated` | Different subjects. There is no shared proposition to agree or disagree about — NLI is being asked a question that has no answer. |

The `agree`/`conflict` line is drawn where the **product** needs it, not where
formal logic does. The disagreement panel exists to stop the generator
silently picking a side when the corpus genuinely disagrees. Two statements
that a knowledgeable reader reconciles in one step are `agree`, even if a
strict reading has them contradicting — surfacing those as "your sources
disagree" is a false alarm, and a false alarm also flips `pipeline.py` into
`CONFLICT_SYSTEM`, which makes the model hedge on sources that were fine.

The hardest case in the set is marked in its note: `init-containers.md` says
init containers *always run to completion*, `sidecar-containers.md` says an
init container with `restartPolicy: Always` *runs for the life of the Pod*.
Propositionally that is a contradiction. It is labelled `agree`, because the
second is the documented special case of the first and no reader is misled.

### `origin`

| value | meaning |
|---|---|
| `corpus` | Both texts are verbatim chunks from the indexed corpus. |
| `perturbed` | One text is a real chunk with a single fact minimally edited to its opposite. `perturbed_from` records the source, `perturbation` describes the edit. |

This distinction is load-bearing. Genuine contradictions are rare here — the
corpus is curated Kubernetes documentation plus chat transcripts, which mostly
agree with themselves. Without perturbed pairs there would be too few
positives to measure recall at all. But their base rate is invented, so:

**Precision is computed over `corpus` rows only. Recall uses everything.**
`calibrate.py` enforces this. A precision number quoted over the whole file
would be fiction.

## Rubric: grounding.jsonl

Label each sentence **against the premises stored in its row**, not against
the corpus as a whole and not against what you happen to know about
Kubernetes.

This is the part that is easy to get wrong. The question being calibrated is
*"given the premises the system retrieved, did it assign the right tag?"* If a
sentence is labelled `grounded` on the strength of a chunk that retrieval
never surfaced, the threshold gets tuned to paper over a retrieval miss, and
the tag it produces will be wrong for a different reason.

| label | meaning |
|---|---|
| `grounded` | A stored premise states this. Quotable — you could point at the sentence in the source. |
| `inferred` | The premises support it but do not state it. A reasonable synthesis, a paraphrase that adds a step, or a claim assembled from two premises. |
| `uncertain` | Nothing in the premises supports it, or a premise contradicts it. |

Set `retrieval_miss: true` when the claim looks true of the corpus but the
*retrieved* premises do not support it. Those rows are still labelled
`uncertain` (that is the correct tag given the premises), and the flag keeps a
count of how much of the grounding error is really a retrieval problem —
a threshold cannot fix those.

`skipped` is not a label. `verify.py` drops non-claims (headings, list
lead-ins) before scoring via `_is_claim`, so they have no tag to calibrate.
The pool marks them `skipped_by_code` for review; they are not carried into
`grounding.jsonl`.

## What the pairs are scored on

`src/nli.py` truncates every premise and hypothesis to `MAX_CHARS = 900`
before scoring, and mean chunk length in this corpus is ~870 characters, so a
minority of chunks are cut. The label files store **full** chunk text; the
truncation happens at scoring time in both production and calibration, so the
two agree.

`_scores.json` is keyed on a hash of the model name plus the *post-truncation*
text, so it invalidates itself correctly: changing `NLI_MODEL` misses every
entry, and raising `MAX_CHARS` misses exactly the rows long enough to have
been cut. Editing a label never invalidates a score. Deleting the file only
costs time.
