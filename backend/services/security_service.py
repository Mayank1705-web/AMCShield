from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

from argon2 import PasswordHasher
from argon2.exceptions import (
    InvalidHashError,
    VerificationError,
    VerifyMismatchError,
)
from fastapi import HTTPException, Request
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..config import settings


# ============================================================
# Password hashing
# ============================================================

password_hasher = PasswordHasher(
    time_cost=3,
    memory_cost=65536,
    parallelism=4,
)


# ============================================================
# Date / time helpers
# ============================================================

def utcnow() -> datetime:
    """
    Return the current UTC time as a timezone-aware datetime.
    """
    return datetime.now(timezone.utc)


def _as_utc(value: datetime | None) -> datetime | None:
    """
    Normalize a datetime to timezone-aware UTC.

    SQLite may return DateTime(timezone=True) values as naive
    datetimes. Treat naive values as UTC so comparisons never
    mix naive and aware datetime objects.
    """
    if value is None:
        return None

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


# ============================================================
# Password validation
# ============================================================

def validate_password(password: str) -> None:
    """
    Validate the AMCShield password policy.

    Requirements:
    - 12 to 128 characters
    - At least three of:
        lowercase
        uppercase
        digit
        special character
    """
    if len(password) < 12:
        raise ValueError(
            "Password must contain at least 12 characters."
        )

    if len(password) > 128:
        raise ValueError(
            "Password must not exceed 128 characters."
        )

    has_lower = any(
        character.islower()
        for character in password
    )

    has_upper = any(
        character.isupper()
        for character in password
    )

    has_digit = any(
        character.isdigit()
        for character in password
    )

    has_special = any(
        not character.isalnum()
        for character in password
    )

    complexity_count = sum(
        [
            has_lower,
            has_upper,
            has_digit,
            has_special,
        ]
    )

    if complexity_count < 3:
        raise ValueError(
            "Password must include at least three of "
            "lowercase, uppercase, number and special character."
        )


def hash_password(password: str) -> str:
    """
    Validate and hash a password using Argon2id
    through argon2-cffi's PasswordHasher.
    """
    validate_password(password)

    return password_hasher.hash(password)


def verify_password(
    password_hash: str,
    password: str,
) -> bool:
    """
    Verify a password against an Argon2 hash.

    Returns False for invalid/malformed hashes or
    password mismatches rather than exposing details.
    """
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


# ============================================================
# Secure token helpers
# ============================================================

def generate_token() -> str:
    """
    Generate a cryptographically secure random token.

    48 random bytes encoded using URL-safe Base64.
    """
    return secrets.token_urlsafe(48)


def hash_token(token: str) -> str:
    """
    SHA-256 hash of a one-time account token.

    Only the hash should be stored in the database.
    """
    return hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()


# ============================================================
# Login rate limiting
# ============================================================

def _rate_limit_key(
    kind: str,
    value: str,
) -> str:
    """
    Create an HMAC-based rate-limit identifier.

    Raw email/username/IP values are never stored as the
    rate-limit database key.
    """
    normalized_value = value.strip().lower()

    message = (
        f"{kind}:{normalized_value}"
    ).encode("utf-8")

    return hmac.new(
        settings.session_secret.encode("utf-8"),
        message,
        hashlib.sha256,
    ).hexdigest()


def _login_keys(
    identity: str,
    ip_address: str | None,
) -> tuple[str, str]:
    """
    Create independent identity and IP buckets.

    This prevents an attacker from bypassing the rate limit
    simply by changing either the identity or source IP.
    """
    identity_key = _rate_limit_key(
        "identity",
        identity,
    )

    ip_key = _rate_limit_key(
        "ip",
        ip_address or "unknown",
    )

    return identity_key, ip_key


def _get_attempt(
    db: Session,
    key: str,
):
    """
    Retrieve a login rate-limit record.
    """
    from ..database.models import LoginAttempt

    return db.scalar(
        select(LoginAttempt).where(
            LoginAttempt.key == key
        )
    )


def _refresh_window(
    row,
    now: datetime,
) -> None:
    """
    Reset an expired login-attempt window.
    """
    window_started_at = _as_utc(
        row.window_started_at
    )

    if window_started_at is None:
        row.window_started_at = now
        row.failed_attempts = 0
        row.locked_until = None
        return

    window_expired = (
        window_started_at
        + timedelta(
            minutes=settings.login_attempt_window_minutes
        )
        <= now
    )

    if window_expired:
        row.window_started_at = now
        row.failed_attempts = 0
        row.locked_until = None


def check_login_rate_limit(
    db: Session,
    identity: str,
    ip_address: str | None,
) -> int:
    """
    Check login rate limiting.

    Two independent buckets are checked:

        1. normalized identity
        2. source IP

    Returns:
        0       -> login may proceed
        > 0     -> remaining lockout seconds
    """
    now = utcnow()

    max_retry = 0
    changed = False

    for key in _login_keys(
        identity,
        ip_address,
    ):
        row = _get_attempt(
            db,
            key,
        )

        if row is None:
            continue

        before = (
            row.failed_attempts,
            row.window_started_at,
            row.locked_until,
        )

        _refresh_window(
            row,
            now,
        )

        locked_until = _as_utc(
            row.locked_until
        )

        after = (
            row.failed_attempts,
            row.window_started_at,
            row.locked_until,
        )

        if before != after:
            changed = True

        if (
            locked_until is not None
            and locked_until > now
        ):
            retry_seconds = int(
                (
                    locked_until - now
                ).total_seconds()
            )

            max_retry = max(
                max_retry,
                retry_seconds,
            )

    if changed:
        db.commit()

    return max_retry


def record_login_failure(
    db: Session,
    identity: str,
    ip_address: str | None,
) -> int:
    """
    Record a failed login attempt in both the identity
    and IP buckets.

    Returns the current lockout duration in seconds.
    """
    from ..database.models import LoginAttempt

    now = utcnow()

    max_retry = 0

    for key in _login_keys(
        identity,
        ip_address,
    ):
        row = _get_attempt(
            db,
            key,
        )

        if row is None:
            row = LoginAttempt(
                key=key,
                failed_attempts=0,
                window_started_at=now,
                locked_until=None,
                updated_at=now,
            )

            db.add(row)
            db.flush()

        _refresh_window(
            row,
            now,
        )

        row.failed_attempts += 1
        row.updated_at = now

        if (
            row.failed_attempts
            >= settings.login_max_attempts
        ):
            row.locked_until = (
                now
                + timedelta(
                    minutes=settings.login_lockout_minutes
                )
            )

        locked_until = _as_utc(
            row.locked_until
        )

        if (
            locked_until is not None
            and locked_until > now
        ):
            retry_seconds = int(
                (
                    locked_until - now
                ).total_seconds()
            )

            max_retry = max(
                max_retry,
                retry_seconds,
            )

    db.commit()

    return max_retry


def clear_login_failures(
    db: Session,
    identity: str,
    ip_address: str | None,
) -> None:
    """
    Clear both identity and IP login-failure buckets
    after successful authentication.
    """
    from ..database.models import LoginAttempt

    keys = _login_keys(
        identity,
        ip_address,
    )

    db.execute(
        delete(LoginAttempt).where(
            LoginAttempt.key.in_(keys)
        )
    )

    db.commit()


# ============================================================
# CSRF protection
# ============================================================

CSRF_SESSION_KEY = "csrf_token"
CSRF_HEADER = "X-CSRF-Token"


def get_csrf_token(
    request: Request,
) -> str:
    """
    Get the CSRF token associated with the current session.

    If the session does not yet contain one, create a
    cryptographically secure token and store it in the
    signed session.
    """
    token = request.session.get(
        CSRF_SESSION_KEY
    )

    if not token:
        token = secrets.token_urlsafe(32)

        request.session[
            CSRF_SESSION_KEY
        ] = token

    return token


def csrf_is_valid(
    request: Request,
) -> bool:
    """
    Validate the CSRF token supplied by the client.

    The token must exist both:
        - in the signed session
        - in the X-CSRF-Token header
    """
    expected = request.session.get(
        CSRF_SESSION_KEY
    )

    supplied = request.headers.get(
        CSRF_HEADER
    )

    if not expected or not supplied:
        return False

    return hmac.compare_digest(
        str(expected),
        str(supplied),
    )


def require_csrf(
    request: Request,
) -> None:
    """
    FastAPI dependency used by state-changing routes.

    Safe methods do not require CSRF.

    Protected methods require:
        X-CSRF-Token
    matching the token stored in the current session.
    """
    if request.method in {
        "GET",
        "HEAD",
        "OPTIONS",
    }:
        return

    if not csrf_is_valid(request):
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing CSRF token.",
        )