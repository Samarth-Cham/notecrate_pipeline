# notecrate_pipeline

Multi-contextual RAG over a private document corpus, served by Llama 3.1 via
Ollama with pgvector for storage.

## Pipeline

```
question
   -> router        classify simple vs multi-hop, decompose into sub-queries
   -> retrieval     pgvector + Postgres full-text, fused with RRF
   -> rerank        bge-reranker cross-encoder, 20 candidates -> top 5,
                    then a role-conditioned boost/penalty
   -> merge         round-robin across sub-queries, deduped
   -> memory        recall relevant prior turns (not the whole transcript)
   -> conflict      pairwise NLI between chunks  [DISABLED - see below]
   -> generate      Llama 3.1 with forced inline citations
   -> verify        per-sentence Grounded / Inferred / Uncertain
```

Everything routes through `answer_question` in [src/pipeline.py](src/pipeline.py),
so the CLI, the API and the eval harness all exercise the same path.

| Module | Role |
|---|---|
| `src/llm.py` | Ollama client — embeddings and chat, one place for model config |
| `src/router.py` | Query classification and multi-hop decomposition |
| `src/hybrid_search.py` | Vector + keyword retrieval, RRF fusion |
| `src/rerank.py` | Cross-encoder reranking |
| `src/roles.py` | Role taxonomy, document tagging, score conditioning |
| `src/backfill_roles.py` | One-off: tag an already-indexed corpus |
| `src/memory.py` | Conversation turns, recalled by relevance |
| `src/nli.py` | Shared NLI cross-encoder (lazy-loaded, used by the two below) |
| `src/conflict.py` | Contradiction detection between chunks (not working) |
| `src/verify.py` | Per-sentence grounding classification |
| `src/pipeline.py` | Orchestration and prompt assembly |
| `src/api.py` | FastAPI surface |

## Running it

Requires Ollama running locally with `llama3.1:8b` and `nomic-embed-text`.

```bash
cp .env.example .env                   # then fill JWT_SECRET and DEMO_USER_PASSWORD
docker compose up -d db
python src/index.py                    # chunk + embed + index the corpus
python src/backfill_roles.py           # tag documents by audience
python src/auth.py seed                # create the demo accounts
python src/ask.py "how does pod restart policy work"
uvicorn src.api:app --reload           # http://localhost:8000/docs
```

Set `OLLAMA_URL` to `127.0.0.1`, not `localhost` — see the note in
`.env.example`; it is worth ~2s on *every* model call.

`index.py` drops and recreates the chunks table, so re-run `backfill_roles.py`
after any re-index.

## Auth, and the two independent mechanisms

`/query` requires a bearer token from `POST /token` (OAuth2 password flow).

**The retrieval role comes from the token, never the request.** There is no
`role` field in `QueryRequest` — that is the point. Plan §2.3 requires role to
derive from the authenticated identity rather than a UI toggle, so a caller
sending `{"role": "senior"}` is simply ignored. `conversation_id` is namespaced
by username server-side, so one user cannot read another's thread by copying
an id.

Role and permission are **separate**, and conflating them is a security bug:

| | mechanism | where | failure mode |
|---|---|---|---|
| `permission_scope` | hard filter | inside the retrieval SQL, pre-rank | data leak |
| `roles` | ±0.08 boost | after reranking | slightly worse ordering |

A junior deliberately still *receives* senior-tagged chunks, ranked lower —
that is what "boost, not filter" means, so `roles` can never be what stands
between a user and a document. `permission_scope` does that job, in the WHERE
clause of both retrieval CTEs, before any `LIMIT`. Filtering afterwards would
mean the reranker, the conflict check and the verification pass had already
read text the user cannot see.

Scopes are threaded through **every** retrieval path, including the
verification pass — which re-retrieves per sentence and would otherwise leak
private text through `support_source`. `answer_question(scopes=...)` is a
required argument with no default, so a new call site cannot silently read the
whole corpus; local tools pass `permissions.UNRESTRICTED` explicitly.

Demo accounts show the two axes are independent: `jamie` is junior/`public`,
`sam` is senior/`public+private`.

```bash
python src/backfill_scopes.py    # assign scopes (re-run after any re-index)
```

`POST /query` returns the answer plus `sources`, `sentences` (inline grounding
tags), a `grounding` summary, `history`, and `timings` (per-stage seconds).

## Role-aware retrieval

Chunks carry an applicable-audience list (`all` / `junior` / `senior`). The
reranker applies a boost on an exact match and a penalty on a mismatch —
never a filter, so a senior asking a beginner question still gets the
beginner document.

`all` is deliberately **not** boosted: it is half the corpus, and boosting it
applies a near-uniform shift, which cannot reorder anything.

```bash
python eval/run_role_eval.py           # same query, both roles, side by side
```

Mean junior/senior retrieval overlap is 0.71 across the sample queries —
4 of 8 unchanged, 0 of 8 fully disjoint. Disjoint would mean the boost had
become a filter.

## Conversation memory

Turns are embedded and stored; each question recalls the most *relevant* prior
turns rather than the most recent, with the latest turn always included so
follow-ups can resolve their antecedent. Verified end-to-end: after a
StatefulSet question, two unrelated turns, and then "what guarantees does it
give about pod identity?", the StatefulSet turn is recalled top (0.57) and the
answer resolves "it" correctly.

## Tests and eval

```bash
pip install -r requirements-dev.txt
pytest tests/                          # deterministic logic; no DB, no models
python eval/run_eval.py reranked       # retrieval metrics
python eval/run_answer_eval.py         # generation metrics (~25 min)
python eval/calibrate.py               # sweep detector thresholds vs labels
```

Latest full run (`eval/results/week5_minilm_*`):

| Metric | Value |
|---|---|
| faithfulness | 0.579 |
| answer relevancy | 0.863 |
| keyword coverage | 0.692 |
| refusal accuracy | 0.75 (3 at gate, 0 at model) |
| latency mean / p95 | 8.3s / 16.7s |

Disabling conflict detection moved faithfulness 0.417 → 0.610 and cut latency
40%. Re-enable it with `--conflicts` to reproduce the cost.

**Aggregate faithfulness has a measured noise floor of about ±0.04** — two runs
of identical code gave 0.429 and 0.393. Per question the swing reaches 0.50.
Compare aggregates over the full set; never single questions.

## Retrieval tuning: why the reranker is a 22M model

The reranker was chosen by measurement, not reputation. `eval/tune_rerank.py`
sweeps model × candidate count against the eval set:

| model | cand | hit@5 | P@5 | R@5 | MRR | rerank |
|---|---|---|---|---|---|---|
| `bge-reranker-base` (278M) | 20 | 1.000 | 0.700 | 0.853 | 0.892 | 7.64s |
| `bge-reranker-base` | 10 | 1.000 | 0.777 | 0.865 | 0.904 | 3.64s |
| `ms-marco-MiniLM-L-6` (22M) | 20 | 1.000 | 0.792 | 0.859 | 0.955 | 1.27s |
| **`ms-marco-MiniLM-L-6`** | **10** | 1.000 | **0.800** | 0.859 | **0.974** | **0.63s** |

**12x faster and better on every metric** — not the trade a 22M model against a
278M one is supposed to produce. Two things explain it. `ms-marco-MiniLM` is
trained directly on passage ranking, which is exactly this task. And
`bge-reranker-base` was *actively harmful*: plain vector search scores MRR
0.973, so at 0.892 the old reranker was reordering good results into worse ones
and charging 7.6s for it. That had been true since Week 3, and no metric being
tracked would have caught it — `hit@5` is 1.000 either way.

End to end this took the full eval from **28.0s to 8.3s mean**.

### What it cost

One regression, and it is a real one. The freshness question — *"How did we fix
the pgvector password authentication issue?"*, whose answer postdates the corpus
— used to be declined by the model:

> "We didn't actually fix a pgvector password authentication issue. The sources
> mention GitHub authentication issues, not pgvector."

It is now answered: *"We used a token (quick fix) or SSH (best fix) [3]."* The
different ranking surfaces GitHub-auth chunks higher, and the model conflates
them with the question. Refusal accuracy fell 1.00 → 0.75.

The noise floor cannot help here: `vec_top` is 0.708, above the 0.69 gate either
way. Freshness questions are on-topic by construction — that is why refusal has
a model layer at all, and the model layer is what regressed.

Kept anyway: a 3.4x latency win with better retrieval metrics and better answer
relevancy, against one question in a four-question negative set. The fix belongs
in the generation prompt (be sceptical when retrieved chunks are topically near
but do not address the question), not in the reranker.

### The coupling that will bite you

**`ROLE_BOOST` must be recalibrated whenever `RERANK_MODEL` changes.** Score
scales differ by an order of magnitude — `bge-reranker` emits sigmoid-squashed
(0,1), `ms-marco-MiniLM` emits raw logits from about −8 to +9. At the old 0.08
the boost became a rounding error, and **nothing would have failed**: the API
keeps returning a `roles` field and an adjustment of no effect.

Scaling by "one rank position" is not enough either. That gave 0.75, and
`eval/run_role_eval.py` showed junior/senior overlap at 0.887 against 0.712
before — a third as effective. MiniLM's gap distribution is skewed (median 2.93,
mean 4.02). Calibrate against the overlap metric instead:

| boost | overlap | unchanged | disjoint |
|---|---|---|---|
| 0.75 | 0.887 | 6/8 | 0/8 |
| 1.50 | 0.815 | 5/8 | 0/8 |
| **2.50** | **0.774** | **4/8** | 0/8 |
| 4.00 | 0.640 | 3/8 | 0/8 |
| 6.00 | 0.640 | 3/8 | 0/8 |

4.00 and 6.00 producing identical results means the boost has stopped competing
with relevance and is simply sorting by role — a filter wearing a boost's
clothes, which §2.3 forbids. 2.50 reproduces the previously calibrated
behaviour with room before that edge.

A test asserts `ROLE_BOOST` matches the active reranker, so a future swap fails
loudly instead of silently disabling the feature.

## Load test (§5)

```bash
docker run --rm -i -e BASE=http://<vm>:8080 -e PASSWORD=... \
  grafana/k6:latest run - < ops/loadtest/query.js
```

Ramps 1 → 5 VUs over 7 minutes against the staging VM (4 cores):

| | avg | p95 |
|---|---|---|
| `stage_retrieve` | 2.12s | 2.82s |
| `stage_generate` | 2.17s | 2.73s |
| `stage_verify` | **6.39s** | **15.76s** |
| end-to-end | 28.8s | 1m27s |

Zero 5xx, and 14% of requests shed as 429 by the rate limiter — which is the
result the limits exist for.

**The finding: stages sum to ~10.7s but requests average 28.8s.** The ~18s gap
is queueing, not work. Capacity is roughly **1–2 concurrent users**; past that,
per-stage timings stay flat and latency is pure CPU contention. There is nothing
to tune away on a single 4-core box — the answer is more cores or more replicas,
and the rate limiter correctly sheds the rest.

`verify` is now 60% of the work, so it is the next target: `RETRIEVE_K = 3 → 1`
in `verify.py` would cut it roughly threefold, and tag accuracy is only ~0.54,
so there is little quality to protect.

## Known limitations

- **Conflict detection is disabled and should stay disabled.** Measured against
  `eval/labels/conflicts.jsonl`, precision is 0.00 at every threshold from 0.05
  to 0.95, under both direction rules. All five genuine corpus conflicts score
  below 0.02. This is not a tuning problem — chunk-level NLI returns confident
  nonsense on unrelated text, and the corpus is mixed-domain. Re-enable with
  `detect_conflicts_enabled=True` only after the detector is rebuilt.
- **Grounding tags are ~50% accurate** (0.500 at the shipped thresholds; the
  best the sweep finds is 0.519). Entailment separates grounded from uncertain
  with AUC 0.598 against 0.5 for a coin flip. The thresholds are already near
  optimal, so the ceiling is the approach, not the constants. Treat the tags as
  indicative, not authoritative.
- **The validation labels are model-generated**, not human. See
  `eval/labels/README.md`. They make the thresholds measurable rather than
  guessed, but they share blind spots with the system they grade.
- **The noise floor margin is 0.026** on 30 questions — tuned, not proven.
  Re-fit when the corpus or the embedding scheme changes; `src/reembed.py`
  prints a reminder for exactly that reason.
- **Freshness questions are answered rather than refused** since the reranker
  swap — see "What it cost" above. One question of four in the negative set,
  and the fix belongs in the generation prompt.
- **Scale-coupled constants.** `ROLE_BOOST` depends on the reranker's score
  scale and `NOISE_FLOOR` on the embedding model. Both fail silently when the
  underlying model changes: retrieval keeps working and the affected feature
  quietly stops. A test guards the first; `reembed.py`'s closing message is
  the only guard on the second.
