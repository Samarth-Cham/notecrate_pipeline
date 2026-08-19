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

Latest full run (`eval/results/week5_final_*`):

| Metric | Value |
|---|---|
| faithfulness | 0.610 |
| answer relevancy | 0.818 |
| keyword coverage | 0.673 |
| refusal accuracy | 1.00 (3 at gate, 1 at model) |
| latency mean / p95 | 28.0s / 41.1s |

Disabling conflict detection moved faithfulness 0.417 → 0.610 and cut latency
40%. Re-enable it with `--conflicts` to reproduce the cost.

**Aggregate faithfulness has a measured noise floor of about ±0.04** — two runs
of identical code gave 0.429 and 0.393. Per question the swing reaches 0.50.
Compare aggregates over the full set; never single questions.

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
- **The noise floor margin is 0.009** on 30 questions — tuned, not proven.
  Re-fit when the corpus changes.
