"""Fernet-based field encryption for sensitive data at rest.

The encryption key is read from ``MIGHTYCODE_ENCRYPTION_KEY`` env var.
Generate one with::

    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""

from __future__ import annotations

import os
from functools import lru_cache

from cryptography.fernet import Fernet


def _get_encryption_key() -> bytes:
    """Read the Fernet key from environment."""
    key = os.environ.get("MIGHTYCODE_ENCRYPTION_KEY", "")
    if not key:
        msg = (
            "MIGHTYCODE_ENCRYPTION_KEY environment variable is not set. "
            "Generate one with: python -c "
            '"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"'
        )
        raise RuntimeError(msg)
    return key.encode()


@lru_cache(maxsize=1)
def _get_fernet() -> Fernet:
    """Return a cached Fernet instance."""
    return Fernet(_get_encryption_key())


def encrypt_value(plaintext: str) -> str:
    """Encrypt a string and return the base64-encoded ciphertext."""
    return _get_fernet().encrypt(plaintext.encode()).decode()


def decrypt_value(ciphertext: str) -> str:
    """Decrypt a base64-encoded ciphertext back to plaintext."""
    return _get_fernet().decrypt(ciphertext.encode()).decode()


def reset_fernet_cache() -> None:
    """Clear the cached Fernet instance (useful for testing)."""
    _get_fernet.cache_clear()
