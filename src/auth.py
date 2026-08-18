"""
JWT authentication, and the authenticated identity that supplies the
retrieval role.

  python src/auth.py seed          # create the users table + demo users
  python src/auth.py hash <pw>     # print a bcrypt hash

Plan section 3.7: the authenticated identity supplies the user's role, which
feeds role-aware retrieval directly — "this is what makes it 'enterprise'
rather than a UI toggle".

That sentence is the whole design constraint. Before this existed, `role` was
a field in the request body, so any caller could claim `senior` by editing
JSON. Role now comes from a signed claim and nothing else: `/query` has no
role parameter left to spoof.
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import bcrypt
import jwt
import psycopg
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.roles import ROLES

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DB_URL = os.environ["DATABASE_URL"]
ALGORITHM = "HS256"
TOKEN_TTL = timedelta(hours=int(os.environ.get("JWT_TTL_HOURS", "12")))

# No default. A signing key that falls back to a constant is not a signing
# key — anyone with the source could mint a `senior` token. Failing loudly at
# import is the correct behaviour for a missing secret.
SECRET_KEY = os.environ.get("JWT_SECRET")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username      text PRIMARY KEY,
    password_hash text NOT NULL,
    role          text NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now()
);
"""


class AuthError(Exception):
    """Credentials missing, invalid, or expired."""


# RFC 7518 §3.2: an HMAC key for HS256 must be at least as long as the hash
# output. A shorter key weakens the signature, which is the only thing
# stopping a caller from minting their own `senior` token.
MIN_SECRET_BYTES = 32


def _require_secret() -> str:
    if not SECRET_KEY:
        raise AuthError(
            "JWT_SECRET is not set. Generate one with:\n"
            "  python -c \"import secrets; print(secrets.token_urlsafe(48))\"\n"
            "and add it to .env"
        )
    if len(SECRET_KEY.encode()) < MIN_SECRET_BYTES:
        raise AuthError(
            f"JWT_SECRET is {len(SECRET_KEY.encode())} bytes; HS256 needs at "
            f"least {MIN_SECRET_BYTES} (RFC 7518 section 3.2)."
        )
    return SECRET_KEY


# --- passwords ---------------------------------------------------------------

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except ValueError:
        # Malformed hash in the database — treat as a failed login, never as
        # a crash that would distinguish "bad hash" from "bad password".
        return False


# --- tokens ------------------------------------------------------------------

def create_token(username: str, role: str) -> str:
    """Sign a token carrying the identity and its role."""
    if role not in ROLES:
        raise AuthError(f"unknown role {role!r}")
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {"sub": username, "role": role, "iat": now, "exp": now + TOKEN_TTL},
        _require_secret(),
        algorithm=ALGORITHM,
    )


def decode_token(token: str) -> dict:
    """Validate a token and return {"username", "role"}.

    Signature and expiry are checked by PyJWT. The role is re-validated here
    as well: a claim is only trustworthy to the extent the value is one we
    recognise, and an unknown role silently disabling the boost would be a
    quiet failure rather than a loud one.
    """
    try:
        claims = jwt.decode(token, _require_secret(), algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError as e:
        raise AuthError("token expired") from e
    except jwt.InvalidTokenError as e:
        raise AuthError("invalid token") from e

    username, role = claims.get("sub"), claims.get("role")
    if not username or role not in ROLES:
        raise AuthError("token is missing a usable identity or role")
    return {"username": username, "role": role}


# --- user store --------------------------------------------------------------

def ensure_schema(conn) -> None:
    conn.execute(SCHEMA)
    conn.commit()


def authenticate(username: str, password: str) -> dict | None:
    """Returns {"username", "role"} on success, None otherwise.

    Deliberately returns the same None for "no such user" and "wrong
    password" so the response cannot be used to enumerate accounts.
    """
    with psycopg.connect(DB_URL) as conn:
        ensure_schema(conn)
        row = conn.execute(
            "SELECT username, password_hash, role FROM users WHERE username = %s",
            (username,),
        ).fetchone()

    if row is None:
        # Spend roughly the same time as a real check, so response latency
        # doesn't leak whether the account exists.
        bcrypt.checkpw(b"timing", bcrypt.hashpw(b"timing", bcrypt.gensalt()))
        return None

    if not verify_password(password, row[1]):
        return None
    return {"username": row[0], "role": row[2]}


def upsert_user(username: str, password: str, role: str) -> None:
    if role not in ROLES:
        raise AuthError(f"unknown role {role!r}; expected one of {list(ROLES)}")
    with psycopg.connect(DB_URL) as conn:
        ensure_schema(conn)
        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s) "
            "ON CONFLICT (username) DO UPDATE SET password_hash = EXCLUDED.password_hash, "
            "role = EXCLUDED.role",
            (username, hash_password(password), role),
        )
        conn.commit()


# Demo accounts. Passwords come from the environment so this file never
# contains a working credential; the seed command refuses to invent one.
DEMO_USERS = [("jamie", "junior"), ("sam", "senior")]


def seed() -> None:
    password = os.environ.get("DEMO_USER_PASSWORD")
    if not password:
        raise AuthError(
            "DEMO_USER_PASSWORD is not set. Choose one and add it to .env — "
            "these are local demo accounts, but a default password committed "
            "to a repo is how demo accounts end up in production."
        )
    for username, role in DEMO_USERS:
        upsert_user(username, password, role)
        print(f"  seeded {username:8s} role={role}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "seed"
    if cmd == "seed":
        print("Seeding demo users...")
        seed()
    elif cmd == "hash" and len(sys.argv) > 2:
        print(hash_password(sys.argv[2]))
    else:
        print(__doc__)
