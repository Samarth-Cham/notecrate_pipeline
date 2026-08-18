"""
Unit tests for the deterministic parts of the pipeline.

Nothing here touches Postgres, Ollama, or a model — these cover the text
handling and merge logic, which is where the subtle bugs live. Retrieval and
generation quality are measured by eval/, not asserted here.

  pytest tests/
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory import as_context
from src.pipeline import _merge, build_prompt
from src.roles import ROLE_BOOST, ROLE_PENALTY, adjust_score, classify_document, matches
from src.verify import (
    _is_claim,
    cited_labels,
    grounding_summary,
    split_sentences,
    strip_citations,
)


# --- sentence segmentation --------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("Pods restart per policy [1]. The default is Always [2].",
     ["Pods restart per policy [1].", "The default is Always [2]."]),
    # an abbreviation's period is not a sentence boundary
    ("Use OnFailure, e.g. for Jobs [1]. Never means never [2].",
     ["Use OnFailure, e.g. for Jobs [1].", "Never means never [2]."]),
    # LLM answers are more list than prose; bullets must not glue together
    ("Options:\n- Always [1]\n- OnFailure [2]",
     ["Options:", "- Always [1]", "- OnFailure [2]"]),
    ("First line.\n\nSecond line.", ["First line.", "Second line."]),
    ("A single clause with no terminator", ["A single clause with no terminator"]),
    ("", []),
])
def test_split_sentences(text, expected):
    assert split_sentences(text) == expected


# --- citation handling ------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("Text [1].", [1]),
    ("Text [1][3].", [1, 3]),
    ("Text [1, 2].", [1, 2]),
    ("A [3] and B [1] and C [3].", [1, 3]),
    ("No markers here.", []),
])
def test_cited_labels(text, expected):
    assert cited_labels(text) == expected


@pytest.mark.parametrize("text, expected", [
    # Regression: the dangling space left by removing "[4]" reads to the NLI
    # model as a sentence boundary and tanked entailment 0.744 -> 0.083.
    ("The Pod is not restarted [4].", "The Pod is not restarted."),
    # An attribution has to be removed together with its marker, or stripping
    # leaves "according to, the Pod stays".
    ("Additionally, according to [3], the Pod stays on the node.",
     "Additionally, the Pod stays on the node."),
    ("As stated in [1], init containers run first.", "init containers run first."),
    ("* The infrastructure container is restarted [2].",
     "The infrastructure container is restarted."),
    ("1. Containers terminate [1].", "Containers terminate."),
    ("A claim [1] [2] holds.", "A claim holds."),
    ("Nothing to strip here.", "Nothing to strip here."),
    # The disagreement panel makes the model attribute positions like this.
    ("* [2] states that init containers run first.", "init containers run first."),
    ("However, [1] explicitly states that sidecars keep running.",
     "However, sidecars keep running."),
])
def test_strip_citations(text, expected):
    assert strip_citations(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("A Pod can restart for these reasons:", False),   # lead-in, asserts nothing
    ("Options include the following:", False),
    ("The Pod infrastructure container is restarted [2].", True),
    ("Yes [1].", False),                                # too short to verify
])
def test_is_claim(text, expected):
    assert _is_claim(text) is expected


# --- grounding summary ------------------------------------------------------

def test_grounding_summary_ignores_skipped():
    sentences = [{"tag": t} for t in
                 ("grounded", "grounded", "inferred", "uncertain", "skipped")]
    assert grounding_summary(sentences) == {
        "grounded": 2, "inferred": 1, "uncertain": 1,
        "classified": 4, "faithfulness": 0.5,
    }


def test_grounding_summary_all_skipped():
    assert grounding_summary([{"tag": "skipped"}])["faithfulness"] is None


# --- multi-hop context merge ------------------------------------------------

def chunk(i):
    return {"id": i, "text": f"chunk {i}", "source": f"s{i}.md", "section": None}


def test_merge_interleaves_sub_queries():
    """A global sort by score would let the stronger sub-query fill the whole
    context — the exact failure decomposition exists to prevent."""
    merged = _merge([[chunk(1), chunk(2), chunk(3)], [chunk(4), chunk(5)]], 4)
    assert [c["id"] for c in merged] == [1, 4, 2, 5]


def test_merge_dedupes_across_sub_queries():
    merged = _merge([[chunk(1), chunk(2)], [chunk(1), chunk(3)]], 4)
    assert [c["id"] for c in merged] == [1, 2, 3]


def test_merge_respects_limit():
    assert len(_merge([[chunk(1), chunk(2), chunk(3)]], 2)) == 2


def test_merge_drains_uneven_lists():
    merged = _merge([[chunk(1)], [chunk(2), chunk(3), chunk(4)]], 10)
    assert [c["id"] for c in merged] == [1, 2, 3, 4]


@pytest.mark.parametrize("lists", [[], [[], []]])
def test_merge_handles_empty(lists):
    assert _merge(lists, 5) == []


# --- prompt assembly --------------------------------------------------------

def test_prompt_without_conflicts_stays_quiet():
    """A standing disagreement instruction makes the model hedge on sources
    that agree, so it must only appear when a conflict was actually found."""
    system, user = build_prompt("q?", [chunk(1), chunk(2)], [])
    assert "Detected contradictions" not in user["content"]
    assert "disagree" not in system["content"]
    assert "[1] s1.md" in user["content"]


def test_prompt_with_conflicts_uses_citation_labels():
    conflicts = [{"a": 0, "b": 1, "score": 0.9,
                  "a_source": "s1.md", "b_source": "s2.md"}]
    system, user = build_prompt("q?", [chunk(1), chunk(2)], conflicts)
    # chunk indices are 0-based internally, citation labels are 1-based
    assert "[1] and [2] appear to contradict" in user["content"]
    assert "disagree" in system["content"]


def test_history_precedes_sources_in_prompt():
    """History is background for interpreting the question, not citable
    material — if it lands after the sources the model cites prior answers
    as though they were corpus documents."""
    history = [{"question": "what is a Pod?", "answer": "A group of containers."}]
    _, user = build_prompt("and how do they restart?", [chunk(1)], [], history)
    assert user["content"].index("Earlier in this conversation") < \
           user["content"].index("Sources:")


# --- role conditioning ------------------------------------------------------

@pytest.mark.parametrize("chunk_roles, user_role, expected", [
    (["junior"], "junior", True),        # exact match
    (["senior"], "junior", False),       # mismatch
    (["junior", "senior"], "senior", True),
    # "all" carries no audience signal, so it must not be boosted: it is half
    # the corpus, and boosting it applies a near-uniform shift that reorders
    # nothing.
    (["all"], "junior", None),
    (["all"], "senior", None),
    (None, "junior", None),              # backfill hasn't run — leave alone
    ([], "junior", None),
    (["junior"], None, None),            # no role requested
])
def test_role_matching(chunk_roles, user_role, expected):
    assert matches(chunk_roles, user_role) is expected


def test_adjust_score_directions():
    assert adjust_score(0.5, ["junior"], "junior") == pytest.approx(0.5 + ROLE_BOOST)
    assert adjust_score(0.5, ["senior"], "junior") == pytest.approx(0.5 - ROLE_PENALTY)
    assert adjust_score(0.5, ["all"], "junior") == 0.5
    assert adjust_score(0.5, None, "junior") == 0.5


def test_role_boost_cannot_outrank_a_clear_relevance_gap():
    """The plan requires a boost, not a filter: a strongly relevant chunk for
    the 'wrong' audience must still beat a weak one for the right audience."""
    relevant_wrong_role = adjust_score(0.95, ["senior"], "junior")
    weak_right_role = adjust_score(0.60, ["junior"], "junior")
    assert relevant_wrong_role > weak_right_role


def test_classify_document_falls_back_to_all(monkeypatch):
    """A classification failure must not silently bias retrieval."""
    monkeypatch.setattr("src.roles.chat_json", lambda *a, **k: None)
    assert classify_document("x.md", "text") == ["all"]
    monkeypatch.setattr("src.roles.chat_json", lambda *a, **k: {"roles": ["bogus"]})
    assert classify_document("x.md", "text") == ["all"]


def test_classify_document_collapses_both_roles_to_all():
    """Tagged for every role says the same thing as "all"; storing it one way
    keeps the match test simple."""
    import src.roles as roles_mod
    original = roles_mod.chat_json
    roles_mod.chat_json = lambda *a, **k: {"roles": ["junior", "senior"]}
    try:
        assert classify_document("x.md", "text") == ["all"]
    finally:
        roles_mod.chat_json = original


# --- conversation memory ----------------------------------------------------

def test_as_context_renders_turns():
    turns = [{"question": "what is a Pod?", "answer": "A group of containers."},
             {"question": "how many?", "answer": "One or more."}]
    rendered = as_context(turns)
    assert "Earlier question: what is a Pod?" in rendered
    assert "Earlier answer: One or more." in rendered


def test_as_context_empty():
    assert as_context([]) == ""
