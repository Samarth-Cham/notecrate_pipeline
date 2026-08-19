"""
Auth unit tests.

No database and no HTTP — these cover token minting and validation, which is
where a mistake is silent and serious. Endpoint wiring is exercised separately
against a running API.
"""

import sys
import time
from datetime import timedelta
from pathlib import Path

import jwt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.auth as auth
from src.auth import (
    AuthError,
    create_token,
    decode_token,
    hash_password,
    verify_password,
)


@pytest.fixture(autouse=True)
def _secret(monkeypatch):
    """Every test signs with a known key, never the developer's real one."""
    monkeypatch.setattr(auth, "SECRET_KEY", "test-secret-not-a-real-key-padded-to-32-bytes-minimum")


# --- passwords --------------------------------------------------------------

def test_password_roundtrip():
    h = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", h)
    assert not verify_password("wrong password", h)


def test_password_hash_is_salted():
    """Two hashes of the same password must differ, or the hashes leak which
    users share a password."""
    assert hash_password("same") != hash_password("same")


def test_verify_password_rejects_malformed_hash():
    """A corrupt row must read as a failed login, not raise — otherwise the
    error distinguishes 'bad hash' from 'bad password'."""
    assert not verify_password("anything", "not-a-bcrypt-hash")


# --- tokens -----------------------------------------------------------------

def test_token_roundtrip():
    claims = decode_token(create_token("jamie", "junior", ["public"]))
    assert claims == {"username": "jamie", "role": "junior", "scopes": ["public"]}


def test_token_rejects_unknown_role_at_minting():
    with pytest.raises(AuthError):
        create_token("jamie", "admin")


def test_token_rejects_tampered_signature():
    """The whole security property: editing the payload must invalidate it."""
    token = create_token("jamie", "junior")
    forged = jwt.encode(
        {**jwt.decode(token, auth.SECRET_KEY, algorithms=["HS256"]), "role": "senior"},
        "a-different-secret-also-padded-to-32-bytes-min",
        algorithm="HS256",
    )
    with pytest.raises(AuthError):
        decode_token(forged)


def test_token_rejects_unsigned_alg_none():
    """`alg: none` is the classic JWT bypass — PyJWT must not accept it
    because we pin algorithms on decode."""
    forged = jwt.encode({"sub": "jamie", "role": "senior"}, key="", algorithm="none")
    with pytest.raises(AuthError):
        decode_token(forged)


def test_token_rejects_valid_signature_with_unknown_role():
    """Correctly signed but claiming a role we do not recognise. An unknown
    role must fail loudly rather than silently disabling role conditioning."""
    forged = jwt.encode({"sub": "jamie", "role": "admin"}, auth.SECRET_KEY,
                        algorithm="HS256")
    with pytest.raises(AuthError):
        decode_token(forged)


def test_token_rejects_missing_subject():
    forged = jwt.encode({"role": "junior"}, auth.SECRET_KEY, algorithm="HS256")
    with pytest.raises(AuthError):
        decode_token(forged)


def test_token_expires(monkeypatch):
    monkeypatch.setattr(auth, "TOKEN_TTL", timedelta(seconds=-1))
    with pytest.raises(AuthError, match="expired"):
        decode_token(create_token("jamie", "junior"))


def test_token_is_not_expired_within_ttl(monkeypatch):
    monkeypatch.setattr(auth, "TOKEN_TTL", timedelta(seconds=5))
    token = create_token("jamie", "junior")
    time.sleep(0.1)
    assert decode_token(token)["username"] == "jamie"


def test_missing_secret_is_an_error_not_a_default(monkeypatch):
    """A signing key that falls back to a constant is not a signing key."""
    monkeypatch.setattr(auth, "SECRET_KEY", None)
    with pytest.raises(AuthError, match="JWT_SECRET"):
        create_token("jamie", "junior")


def test_short_secret_is_rejected(monkeypatch):
    """RFC 7518 section 3.2: HS256 needs >= 32 bytes of key."""
    monkeypatch.setattr(auth, "SECRET_KEY", "too-short")
    with pytest.raises(AuthError, match="at least 32"):
        create_token("jamie", "junior")


# --- permission scopes ------------------------------------------------------

def test_token_carries_scopes():
    claims = decode_token(create_token("sam", "senior", ["public", "private"]))
    assert claims["scopes"] == ["public", "private"]


def test_token_rejects_unknown_scope_at_minting():
    with pytest.raises(AuthError, match="unknown scopes"):
        create_token("sam", "senior", ["public", "everything"])


def test_decode_drops_unrecognised_scopes():
    """A stale claim must narrow access, never widen it or lock the user out."""
    forged = jwt.encode({"sub": "sam", "role": "senior",
                         "scopes": ["public", "legacy-admin"]},
                        auth.SECRET_KEY, algorithm="HS256")
    assert decode_token(forged)["scopes"] == ["public"]


def test_missing_scopes_claim_fails_closed():
    """A token with no scopes must retrieve nothing, not everything."""
    forged = jwt.encode({"sub": "sam", "role": "senior"},
                        auth.SECRET_KEY, algorithm="HS256")
    assert decode_token(forged)["scopes"] == []
