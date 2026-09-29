import hashlib
import secrets

from argon2 import PasswordHasher
from argon2.exceptions import (
    InvalidHashError,
    VerificationError,
    VerifyMismatchError,
)


password_hasher = PasswordHasher(
    time_cost=3,
    memory_cost=65536,
    parallelism=4,
)


def validate_password(password: str):

    if len(password) < 12:
        raise ValueError(
            "Password must contain at least 12 characters."
        )

    if len(password) > 128:
        raise ValueError(
            "Password must not exceed 128 characters."
        )

    has_lower = any(
        c.islower()
        for c in password
    )

    has_upper = any(
        c.isupper()
        for c in password
    )

    has_digit = any(
        c.isdigit()
        for c in password
    )

    has_special = any(
        not c.isalnum()
        for c in password
    )

    if sum(
        [
            has_lower,
            has_upper,
            has_digit,
            has_special,
        ]
    ) < 3:

        raise ValueError(
            "Password must include at least three of "
            "lowercase, uppercase, number and special character."
        )


def hash_password(password: str) -> str:

    validate_password(password)

    return password_hasher.hash(password)


def verify_password(
    password_hash: str,
    password: str,
) -> bool:

    try:

        return password_hasher.verify(
            password_hash,
            password,
        )

    except (
        VerifyMismatchError,
        VerificationError,
        InvalidHashError,
    ):

        return False


def generate_token() -> str:

    return secrets.token_urlsafe(48)


def hash_token(token: str) -> str:

    return hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()