"""
Permission scope logic.

The SQL filter itself is exercised against a live database in the end-to-end
checks; these cover the normalisation that decides what the SQL is asked for,
which is where a fail-open bug would hide.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.permissions import (
    PRIVATE,
    PUBLIC,
    UNRESTRICTED,
    normalise,
    scope_for_source_type,
)


@pytest.mark.parametrize("source_type, expected", [
    ("chat_export", PRIVATE),
    ("markdown", PUBLIC),
    ("pdf", PUBLIC),
    (None, PUBLIC),
])
def test_scope_for_source_type(source_type, expected):
    assert scope_for_source_type(source_type) == expected


def test_unrestricted_means_no_filter():
    """None is what the SQL reads as 'skip the predicate'."""
    assert normalise(UNRESTRICTED) is None


@pytest.mark.parametrize("scopes", [[], None])
def test_empty_scopes_fail_closed(scopes):
    """The critical case: no scopes must produce an empty list (matches no
    rows), never None (matches every row)."""
    assert normalise(scopes) == []


def test_unknown_scopes_are_dropped_not_passed_through():
    """A malformed claim narrows access instead of widening it."""
    assert normalise(["public", "root", "*"]) == [PUBLIC]


def test_known_scopes_survive():
    assert sorted(normalise([PUBLIC, PRIVATE])) == [PRIVATE, PUBLIC]


def test_all_unknown_scopes_fail_closed():
    assert normalise(["admin", "root"]) == []
