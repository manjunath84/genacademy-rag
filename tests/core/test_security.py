from genacademy_rag.core.security import (
    hash_password,
    is_bcrypt_hash,
    new_invite_code,
    split_invite_code,
    verify_password,
    verify_secret,
)


def test_password_hash_round_trip_and_wrong_password_fails():
    hashed = hash_password("correct horse battery staple")
    assert hashed != "correct horse battery staple"
    assert is_bcrypt_hash(hashed)
    assert verify_password("correct horse battery staple", hashed)
    assert not verify_password("wrong", hashed)


def test_long_password_is_supported_before_bcrypt_limit():
    long_password = "x" * 200
    hashed = hash_password(long_password)
    assert verify_password(long_password, hashed)
    assert not verify_password(long_password + "y", hashed)


def test_invite_code_is_structured_and_secret_hash_verifies():
    code = new_invite_code()
    code_id, secret, secret_hash = code
    raw_code = f"{code_id}.{secret}"
    assert code.id == code_id
    assert code.secret == secret
    assert code.secret_hash == secret_hash
    assert "." in raw_code
    assert secret not in secret_hash
    assert is_bcrypt_hash(secret_hash)
    assert split_invite_code(raw_code) == (code_id, secret)
    assert verify_secret(secret, secret_hash)
    assert not verify_secret("wrong-secret", secret_hash)


def test_malformed_invite_code_returns_none():
    assert split_invite_code("") is None
    assert split_invite_code("missing-dot") is None
    assert split_invite_code(".missing-id") is None
    assert split_invite_code("missing-secret.") is None
