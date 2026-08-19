"""
Permission scopes: access control, as distinct from role relevance.

  python src/backfill_scopes.py       # assign scopes to an indexed corpus

These are the two halves of "multi-contextual", and confusing them is a
security bug rather than a quality one:

  permission_scope   HARD filter, applied inside the retrieval SQL before
                     ranking. Determines what a user may see at all.
                     Failure mode: a data leak.

  roles (roles.py)   SOFT boost, +/-0.08 after reranking. Determines what a
                     user sees FIRST among things they may already see.
                     Failure mode: slightly worse ordering.

A junior deliberately still receives senior-tagged chunks, ranked lower —
plan section 2.3 requires a boost, not a filter. So `roles` can never be the
thing standing between a user and a document they are not cleared for. That
job belongs here.

Two rules this module exists to enforce:

1. The filter runs in SQL, pre-rank. Not in Python afterwards. Filtering
   after retrieval means the reranker, the NLI conflict check and the
   verification pass have all already read text the user cannot see, and any
   of them can leak it through a citation, a support snippet or an error.

2. It fails closed. A token with no scopes retrieves nothing, not everything.
"""

# The sandbox corpus splits cleanly: Kubernetes documentation and reference
# PDFs are shareable, exported chat transcripts are personal. Real deployments
# derive this at ingestion from the source's own ACL (a channel id, a drive
# folder, a repo), which is why schema.sql calls it "inherited from the
# invitation" rather than something we compute.
PUBLIC = "public"
PRIVATE = "private"
SCOPES = (PUBLIC, PRIVATE)

# Explicit opt-out for local tools (CLI, eval harnesses) that legitimately
# read the whole corpus. Named rather than `None` so that bypassing the
# permission filter is greppable and can never be a typo — every serving path
# passes a real list instead.
UNRESTRICTED = "__unrestricted__"


def scope_for_source_type(source_type: str | None) -> str:
    """Scope a chunk inherits from where it came from."""
    return PRIVATE if source_type == "chat_export" else PUBLIC


def normalise(scopes) -> list[str] | None:
    """Coerce a caller's scopes into a SQL-ready list.

    Returns None for UNRESTRICTED (meaning "no filter"), otherwise a list of
    recognised scopes. Unknown values are dropped rather than passed through,
    so a malformed token claim narrows access instead of widening it.
    """
    if scopes is UNRESTRICTED or scopes == UNRESTRICTED:
        return None
    if not scopes:
        return []          # fail closed: matches no rows
    return [s for s in scopes if s in SCOPES]
