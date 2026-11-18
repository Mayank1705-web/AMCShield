from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from .database.db import get_db
from .database.models import User, UserRole, UserStatus


def _db_user_from_session(request: Request, db: Session) -> User | None:
    session_user = request.session.get("user")
    if not session_user:
        return None

    if session_user.get("id") == "local-admin":
        return None

    provider = session_user.get("provider")
    user_id = session_user.get("id")

    if provider not in {"local", "github"} or not isinstance(user_id, int):
        return None

    user = db.get(User, user_id)
    if not user or user.status != UserStatus.ACTIVE.value:
        return None

    return user


def is_admin_role(user: dict | None) -> bool:
    if not user:
        return False
    role = str(
        user.get("role_key") or user.get("role") or ""
    ).strip().lower()
    return role in {"admin", "administrator"}


def is_admin_session(request: Request, db: Session) -> bool:
    session_user = request.session.get("user")
    if not session_user or not is_admin_role(session_user):
        return False

    if session_user.get("id") == "local-admin":
        return True

    if session_user.get("provider") != "local" or not isinstance(
        session_user.get("id"), int
    ):
        return False

    user = db.get(User, session_user["id"])
    return bool(
        user
        and user.status == UserStatus.ACTIVE.value
        and user.role == UserRole.ADMIN.value
    )


def require_authenticated(
    request: Request,
    db: Session = Depends(get_db),
) -> dict:
    session_user = request.session.get("user")
    if not session_user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required.",
        )

    if (
        session_user.get("provider") == "local"
        and session_user.get("id") == "local-admin"
    ):
        return session_user

    db_user = _db_user_from_session(request, db)
    if not db_user:
        # Remove a stale/disabled session so it cannot be reused.
        request.session.clear()
        raise HTTPException(
            status_code=401,
            detail="Authentication required.",
        )

    # Rebuild authorization fields from the database on every protected request.
    # This prevents stale/tampered session role data from granting privileges.
    session_user["id"] = db_user.id
    session_user["name"] = db_user.name
    session_user["email"] = db_user.email
    session_user["role_key"] = db_user.role
    session_user["role"] = (
        "Administrator"
        if db_user.role == UserRole.ADMIN.value
        else "User"
    )
    return session_user


def require_admin(
    request: Request,
    db: Session = Depends(get_db),
) -> dict:
    user = request.session.get("user")
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required.",
        )

    if not is_admin_session(request, db):
        raise HTTPException(
            status_code=403,
            detail="Administrator access required.",
        )

    return user
