"""
FastAPI wrapper around the RAG pipeline.

Run:    uvicorn src.api:app --reload
Docs:   http://localhost:8000/docs

/query returns everything the UI needs to render the Week 4 features
without a second call: numbered sources for citations, `conflicts` for the
"sources disagree" panel, and `sentences` for inline grounding tags.

/query requires a bearer token from /token. The retrieval role is read from
that token, never from the request — see src/auth.py.
"""

from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

from src.auth import AuthError, authenticate, create_token, decode_token
from src.pipeline import CONFLICT_DETECTION_ENABLED, answer_question
from src.verify import grounding_summary

app = FastAPI(title="NoteCrate Pipeline", version="0.3.0")

oauth2 = OAuth2PasswordBearer(tokenUrl="token")


def current_user(token: Annotated[str, Depends(oauth2)]) -> dict:
    """Resolve the bearer token to {"username", "role"} or reject the request."""
    try:
        return decode_token(token)
    except AuthError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


CurrentUser = Annotated[dict, Depends(current_user)]

# ---- request / response shapes (the API contract) ---------------------------

class QueryRequest(BaseModel):
    question: str
    verify: bool = True     # skip to trade grounding tags for latency
    route: bool = True      # skip to force single-query retrieval
    # NOTE: there is deliberately no `role` field. It used to live here, which
    # meant any caller could claim `senior` by editing JSON. Role now comes
    # from the signed token and there is nothing left to spoof (plan 2.3/3.7).
    #
    # Conversation thread id. Namespaced server-side by the authenticated
    # username, so passing someone else's id cannot read their history.
    conversation_id: str | None = None
    # Off by default because the detector has 0.00 measured precision (see
    # src/conflict.py). Exposed so the disagreement panel can still be
    # demonstrated, and so its cost stays measurable — enabling it moved
    # faithfulness 0.610 -> 0.417 and latency 28s -> 46s across the eval set.
    detect_conflicts: bool = CONFLICT_DETECTION_ENABLED

class SourceOut(BaseModel):
    label: int
    source: str
    section: str | None
    rerank_score: float          # after the role boost/penalty
    relevance_score: float       # before it — the two differ only when role is set
    roles: list[str] | None
    vec_score: float | None

class HistoryOut(BaseModel):
    """A prior turn recalled by relevance, not recency."""
    question: str
    relevance: float | None

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

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str               # echoed so the UI can label itself; the token is authoritative
    scopes: list[str]

class QueryResponse(BaseModel):
    answer: str
    confidence: str         # "strong" | "weak"
    username: str
    role: str | None
    route: RouteOut
    sources: list[SourceOut]
    conflicts: list[ConflictOut]
    sentences: list[SentenceOut]
    grounding: dict | None
    history: list[HistoryOut]
    timings: dict[str, float]     # seconds per pipeline stage

# ---- endpoints ----------------------------------------------------------------

@app.get("/health")
def health():
    """Unauthenticated on purpose — container healthchecks poll this."""
    return {"status": "ok"}

@app.post("/token", response_model=TokenResponse)
def token(form: Annotated[OAuth2PasswordRequestForm, Depends()]):
    """OAuth2 password flow (plan section 4: FastAPI + OAuth2 password/JWT)."""
    user = authenticate(form.username, form.password)
    if user is None:
        # One message for both "no such user" and "wrong password", so the
        # response cannot be used to enumerate accounts.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return TokenResponse(
        access_token=create_token(user["username"], user["role"], user["scopes"]),
        username=user["username"],
        role=user["role"],
        # Hard permission filter, applied inside the retrieval SQL. Distinct
        # from `role`, which only re-orders what is already visible.
        scopes=user["scopes"],
    )

@app.get("/me")
def me(user: CurrentUser):
    """Who the current token says you are. Lets the UI restore a session
    without re-prompting, and makes an expired token obvious."""
    return user

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, user: CurrentUser):   # plain def, NOT async — blocking I/O
    result = answer_question(
        req.question,
        use_router=req.route,
        verify_answer=req.verify,
        # Role comes from the validated token. This is the line plan 2.3 is
        # about: the same corpus, retrieved differently per identity, with
        # nothing client-supplied in the decision.
        role=user["role"],
        # Hard permission filter, applied inside the retrieval SQL. Distinct
        # from `role`, which only re-orders what is already visible.
        scopes=user["scopes"],
        # Namespaced by username so one user cannot read another's thread by
        # guessing or copying a conversation id.
        conversation_id=(f"{user['username']}:{req.conversation_id}"
                         if req.conversation_id else None),
        detect_conflicts_enabled=req.detect_conflicts,
    )

    if result["refused"]:
        raise HTTPException(status_code=404, detail=result["reason"])

    return QueryResponse(
        answer=result["answer"],
        confidence=result["confidence"],
        username=user["username"],
        role=result["role"],
        route=RouteOut(**result["route"]),
        sources=[
            SourceOut(
                label=i,
                source=c["source"],
                section=c.get("section") or None,
                rerank_score=round(c["rerank_score"], 3),
                relevance_score=round(c["relevance_score"], 3),
                roles=c.get("roles"),
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
        history=[HistoryOut(question=h["question"], relevance=h["relevance"])
                 for h in result["history"]],
        timings=result["timings"],
    )
