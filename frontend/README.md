# frontend

React + Tailwind UI for the NoteCrate pipeline (plan §4 tech stack, §3.3
containerization).

```bash
npm install
npm run dev          # http://localhost:5173, proxies /api -> :8000
```

Point the proxy elsewhere with `VITE_API_TARGET=http://localhost:8001 npm run dev`.

The app **always** calls the API on a relative `/api` path — Vite proxies in
dev, Nginx proxies in the container. Same origin either way, so there is no
CORS configuration anywhere in this project.

## What it renders

| Feature | Plan week |
|---|---|
| Chat, answer, numbered citations linked to sources | 2 |
| Per-sentence Grounded / Inferred / Uncertain tags | 4 |
| "Sources disagree" panel | 4 |
| Role selector, with the score adjustment shown per source | 5 |
| Multi-turn memory, showing which prior turns were recalled | 5 |

Citation chips scroll to and highlight their source. Sentences the pipeline
tags `skipped` (list lead-ins, headings) get no badge — they assert nothing.

## Two deliberate choices

**Grounding tags are shown, with an escape hatch.** Tag accuracy is ~0.54 on
the labelled set, so "Show plain answer" toggles them off. They are useful as a
reading aid and not yet trustworthy as a verdict.

**The disagreement panel states its own unreliability.** Conflict detection has
0.00 measured precision and is off by default; the toggle is labelled
`(broken)`, and the panel carries the number inline. A disagreement claim the
reader cannot calibrate is worse than no claim, and this is a demo surface —
someone will turn it on.

## Container

```bash
docker compose up -d --build frontend    # http://localhost:5173
```

Multi-stage build: `node:22-alpine` compiles, `nginx:1.27-alpine` serves. The
node toolchain never reaches the runtime image.
