"""Thin session auth. Phase 0: 2 seeded users, plaintext dev passwords, session cookie holds email.
RBAC + hashing + invite-code = Phase 1."""
from __future__ import annotations

from genacademy_rag.data.datastore import SQLiteDatastore


def authenticate(datastore: SQLiteDatastore, email: str, password: str) -> dict | None:
    user = datastore.get_user_by_email(email)
    if user and user["password"] == password:
        return user
    return None
