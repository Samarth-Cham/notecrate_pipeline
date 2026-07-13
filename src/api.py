"""
FastAPI wrapper around the RAG pipeline.

Run:    uvicorn src.api:app --reload
Docs:   http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.ask import NOISE_FLOOR, STRONG, build_prompt, generate, retrieve

app = FastAPI(title="NoteCrate Pipeline", version="0.1.0")

# ---- request / response shapes (the API contract) ---------------------------

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5          # optional, defaults if client omits it

class SourceOut(BaseModel):
    label: int
    source: str
    section: str | None
    source_type: str | None
    score: float

class QueryResponse(BaseModel):
    answer: str
    confidence: str          # "strong" | "weak"
    sources: list[SourceOut]

# ---- endpoints ----------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):          # plain def, NOT async — blocking I/O inside
    chunks = retrieve(req.question)

    best = chunks[0]["score"] if chunks else 0.0
    if best < NOISE_FLOOR:
        raise HTTPException(
            status_code=404,
            detail=f"Nothing in the corpus covers this (best match {best:.3f}).",
        )

    answer = generate(build_prompt(req.question, chunks))

    return QueryResponse(
        answer=answer,
        confidence="strong" if best >= STRONG else "weak",
        sources=[
            SourceOut(
                label=i,
                source=c["source"],
                section=c["section"] or None,
                source_type=c.get("source_type"),
                score=round(c["score"], 3),
            )
            for i, c in enumerate(chunks, 1)
        ],
    )