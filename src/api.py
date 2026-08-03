"""
FastAPI wrapper around the RAG pipeline.

Run:    uvicorn src.api:app --reload
Docs:   http://localhost:8000/docs

/query returns everything the UI needs to render the Week 4 features
without a second call: numbered sources for citations, `conflicts` for the
"sources disagree" panel, and `sentences` for inline grounding tags.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.pipeline import answer_question
from src.verify import grounding_summary

app = FastAPI(title="NoteCrate Pipeline", version="0.2.0")

# ---- request / response shapes (the API contract) ---------------------------

class QueryRequest(BaseModel):
    question: str
    verify: bool = True     # skip to trade grounding tags for latency
    route: bool = True      # skip to force single-query retrieval

class SourceOut(BaseModel):
    label: int
    source: str
    section: str | None
    rerank_score: float
    vec_score: float | None

class ConflictOut(BaseModel):
    """A detected contradiction between two sources, by citation label."""
    a: int                  # 1-based citation labels, matching SourceOut.label
    b: int
    a_source: str
    b_source: str
    score: float

class SentenceOut(BaseModel):
    text: str
    tag: str                # "grounded" | "inferred" | "uncertain" | "skipped"
    cited: list[int]
    entailment: float | None
    support_source: str | None

class RouteOut(BaseModel):
    kind: str               # "simple" | "multi_hop"
    sub_queries: list[str]

class QueryResponse(BaseModel):
    answer: str
    confidence: str         # "strong" | "weak"
    route: RouteOut
    sources: list[SourceOut]
    conflicts: list[ConflictOut]
    sentences: list[SentenceOut]
    grounding: dict | None

# ---- endpoints ----------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):          # plain def, NOT async — blocking I/O inside
    result = answer_question(req.question, use_router=req.route,
                             verify_answer=req.verify)

    if result["refused"]:
        raise HTTPException(status_code=404, detail=result["reason"])

    return QueryResponse(
        answer=result["answer"],
        confidence=result["confidence"],
        route=RouteOut(**result["route"]),
        sources=[
            SourceOut(
                label=i,
                source=c["source"],
                section=c.get("section") or None,
                rerank_score=round(c["rerank_score"], 3),
                vec_score=round(c["vec_score"], 3) if c.get("vec_score") is not None else None,
            )
            for i, c in enumerate(result["chunks"], 1)
        ],
        conflicts=[ConflictOut(**{**f, "a": f["a"] + 1, "b": f["b"] + 1,
                                  "score": round(f["score"], 3)})
                   for f in result["conflicts"]],
        sentences=[
            SentenceOut(
                text=s["text"],
                tag=s["tag"],
                cited=s["cited"],
                entailment=s["entailment"],
                support_source=(s["support"] or {}).get("source"),
            )
            for s in result["sentences"]
        ],
        grounding=grounding_summary(result["sentences"]) if result["sentences"] else None,
    )
