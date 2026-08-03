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

from src.pipeline import _merge, build_prompt
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
