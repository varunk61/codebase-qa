"""In-memory user store and account lifecycle."""

from hashing import generate_salt, hash_password, verify_password

_USERS = {}
_DEFAULT_ROLES = ["member"]


def register_user(username, password):
    """Create a new user record with a freshly salted password hash."""
    if username in _USERS:
        raise ValueError("username already taken")
    salt = generate_salt()
    _USERS[username] = {
        "salt": salt,
        "pw_hash": hash_password(password, salt),
        "roles": list(_DEFAULT_ROLES),
        "active": True,
    }
    return _USERS[username]


def get_user(username):
    """Return the stored user record for username, or None."""
    return _USERS.get(username)


def user_exists(username):
    """True when a record exists for username."""
    return username in _USERS


def change_password(username, old_password, new_password):
    """Rotate a user's password after verifying the current one."""
    user = get_user(username)
    if user is None:
        raise KeyError(username)
    if not verify_password(old_password, user["salt"], user["pw_hash"]):
        raise PermissionError("current password does not match")
    salt = generate_salt()
    user["salt"] = salt
    user["pw_hash"] = hash_password(new_password, salt)


def grant_role(username, role):
    """Add a role to an existing user's role list."""
    user = get_user(username)
    if user is None:
        raise KeyError(username)
    if role not in user["roles"]:
        user["roles"].append(role)


def revoke_role(username, role):
    """Remove a role from a user if present."""
    user = get_user(username)
    if user is None:
        raise KeyError(username)
    if role in user["roles"]:
        user["roles"].remove(role)


def has_role(username, role):
    """True when the user holds the named role."""
    user = get_user(username)
    return bool(user) and role in user["roles"]


def deactivate_user(username):
    """Mark an account inactive without deleting its record."""
    user = get_user(username)
    if user is not None:
        user["active"] = False
