"""Password and token hashing primitives."""

import hashlib
import hmac
import os
import base64

_ITERATIONS = 120_000
_SALT_BYTES = 16
_ALGO = "sha256"


def generate_salt(n=_SALT_BYTES):
    """Return n cryptographically-random bytes rendered as a hex string."""
    return os.urandom(n).hex()


def hash_password(password, salt):
    """Derive a PBKDF2-HMAC-SHA256 hash of password using the given salt.

    The work factor is fixed at 120k iterations. Returns a hex digest.
    """
    derived = hashlib.pbkdf2_hmac(
        _ALGO, password.encode(), salt.encode(), _ITERATIONS
    )
    return derived.hex()


def verify_password(password, salt, expected_hash):
    """Constant-time check of a freshly derived hash against expected_hash."""
    candidate = hash_password(password, salt)
    return hmac.compare_digest(candidate, expected_hash)


def needs_rehash(stored_iterations):
    """True when a stored hash used fewer iterations than the current policy."""
    return stored_iterations < _ITERATIONS


def hash_token(token):
    """One-way hash of an opaque token for safe storage in the session table."""
    return hashlib.sha256(token.encode()).hexdigest()


def constant_time_equals(a, b):
    """Wrapper around hmac.compare_digest that accepts str or bytes."""
    if isinstance(a, str):
        a = a.encode()
    if isinstance(b, str):
        b = b.encode()
    return hmac.compare_digest(a, b)


def derive_key(password, salt, length=32):
    """Derive a raw key of `length` bytes for symmetric encryption use."""
    return hashlib.pbkdf2_hmac(_ALGO, password.encode(), salt.encode(),
                               _ITERATIONS, dklen=length)


def encode_hash(salt, digest):
    """Pack salt and digest into a single portable `salt$digest` field."""
    blob = f"{salt}${digest}".encode()
    return base64.urlsafe_b64encode(blob).decode()


def decode_hash(encoded):
    """Split a packed `salt$digest` field back into its two parts."""
    blob = base64.urlsafe_b64decode(encoded.encode()).decode()
    salt, digest = blob.split("$", 1)
    return salt, digest
