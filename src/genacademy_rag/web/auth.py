"""Thin session auth. Credential verification delegates to pure core security."""
from __future__ import annotations

from genacademy_rag.core.security import verify_password
from genacademy_rag.data.datastore import SQLiteDatastore


def authenticate(datastore: SQLiteDatastore, email: str, password: str) -> dict | None:
    user = datastore.get_user_by_email(email)
    if user and verify_password(password, user["password"]):
        return user
    return None
