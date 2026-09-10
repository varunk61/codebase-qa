"""Opaque session tokens with a sliding expiry."""

import secrets
import time

from hashing import hash_token

SESSION_TTL = 3600
_SESSIONS = {}


def create_session(username):
    """Mint a random URL-safe session token bound to username.

    Only the hash of the token is stored, so a leak of the session table
    does not expose usable tokens.
    """
    token = secrets.token_urlsafe(32)
    _SESSIONS[hash_token(token)] = {
        "user": username,
        "created": time.time(),
        "last_seen": time.time(),
    }
    return token


def resolve_session(token):
    """Return the username for a live token, or None if missing or expired."""
    entry = _SESSIONS.get(hash_token(token))
    if entry is None:
        return None
    if time.time() - entry["created"] > SESSION_TTL:
        del _SESSIONS[hash_token(token)]
        return None
    entry["last_seen"] = time.time()
    return entry["user"]


def touch_session(token):
    """Extend a session by resetting its creation timestamp."""
    entry = _SESSIONS.get(hash_token(token))
    if entry is not None:
        entry["created"] = time.time()


def revoke_session(token):
    """Invalidate a single session token immediately."""
    _SESSIONS.pop(hash_token(token), None)


def revoke_all_for_user(username):
    """Drop every session belonging to username (e.g. on password change)."""
    dead = [k for k, v in _SESSIONS.items() if v["user"] == username]
    for k in dead:
        del _SESSIONS[k]


def active_session_count():
    """Number of sessions currently held in memory."""
    return len(_SESSIONS)


def gc_expired():
    """Purge every expired session; return how many were removed."""
    now = time.time()
    dead = [k for k, v in _SESSIONS.items() if now - v["created"] > SESSION_TTL]
    for k in dead:
        del _SESSIONS[k]
    return len(dead)
