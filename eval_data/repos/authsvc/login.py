"""Login / logout flow tying users, hashing and sessions together."""

from hashing import verify_password
from sessions import create_session, resolve_session, revoke_session
from users import get_user, has_role

_FAILED_ATTEMPTS = {}
_MAX_ATTEMPTS = 5


def authenticate(username, password):
    """Return True only if password matches the stored hash for username."""
    user = get_user(username)
    if user is None:
        return False
    if not user.get("active", True):
        return False
    return verify_password(password, user["salt"], user["pw_hash"])


def record_failure(username):
    """Increment the failed-attempt counter for username."""
    _FAILED_ATTEMPTS[username] = _FAILED_ATTEMPTS.get(username, 0) + 1
    return _FAILED_ATTEMPTS[username]


def is_locked_out(username):
    """True once a user has exceeded the allowed number of failed attempts."""
    return _FAILED_ATTEMPTS.get(username, 0) >= _MAX_ATTEMPTS


def login(username, password):
    """Authenticate the credentials and, on success, open a session token."""
    if is_locked_out(username):
        raise PermissionError("account temporarily locked")
    if not authenticate(username, password):
        record_failure(username)
        raise PermissionError("bad credentials")
    _FAILED_ATTEMPTS.pop(username, None)
    return create_session(username)


def logout(token):
    """Revoke the session behind a token."""
    revoke_session(token)


def current_user(token):
    """Resolve the username for a session token, or None."""
    return resolve_session(token)


def require_role(token, role):
    """Return the username only if the session holds the required role."""
    username = resolve_session(token)
    if username is None:
        raise PermissionError("not signed in")
    if not has_role(username, role):
        raise PermissionError(f"missing role: {role}")
    return username
