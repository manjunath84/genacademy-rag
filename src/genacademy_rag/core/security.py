"""Pure security helpers for passwords and invite-code bearer secrets."""
from __future__ import annotations

import base64
import hashlib
import secrets
from typing import NamedTuple

import bcrypt

BCRYPT_PREFIXES = ("$2a$", "$2b$", "$2y$")


class NewInviteCode(NamedTuple):
    id: str
    secret: str
    secret_hash: str


def _bcrypt_input(value: str) -> bytes:
    raw = value.encode("utf-8")
    if len(raw) <= 72:
        return raw
    return base64.b64encode(hashlib.sha256(raw).digest())


def is_bcrypt_hash(value: str | None) -> bool:
    return bool(value and value.startswith(BCRYPT_PREFIXES))


def hash_secret(secret: str) -> str:
    return bcrypt.hashpw(_bcrypt_input(secret), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_secret(secret: str, secret_hash: str) -> bool:
    if not is_bcrypt_hash(secret_hash):
        return False
    try:
        return bcrypt.checkpw(_bcrypt_input(secret), secret_hash.encode("utf-8"))
    except ValueError:
        return False


def hash_password(password: str) -> str:
    return hash_secret(password)


def verify_password(password: str, password_hash: str) -> bool:
    return verify_secret(password, password_hash)


def new_invite_code() -> NewInviteCode:
    code_id = secrets.token_urlsafe(8)
    secret = secrets.token_urlsafe(24)
    return NewInviteCode(id=code_id, secret=secret, secret_hash=hash_secret(secret))


def split_invite_code(raw_code: str) -> tuple[str, str] | None:
    code_id, sep, secret = raw_code.rpartition(".")
    if sep != "." or not code_id or not secret:
        return None
    return code_id, secret
