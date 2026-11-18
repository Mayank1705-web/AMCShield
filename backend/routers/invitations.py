from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ..database.db import get_db
from ..database.models import (
    AccountToken,
    User,
    UserStatus,
    utcnow,
)
from ..services.security_service import (
    hash_password,
    hash_token,
    require_csrf,
    validate_password,
)


router = APIRouter(
    prefix="/account/invite",
    tags=["Account Invitation"],
)


class CompleteInvitationRequest(BaseModel):
    token: str = Field(min_length=20)
    password: str = Field(
        min_length=12,
        max_length=128,
    )
    confirm_password: str = Field(
        min_length=12,
        max_length=128,
    )


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


def get_invitation(
    db: Session,
    raw_token: str,
):
    """
    Resolve an unused invite/password-change token.

    Supported purposes:
        invite
        password_change
    """

    token_hash = hash_token(raw_token)

    record = db.scalar(
        select(AccountToken).where(
            AccountToken.token_hash == token_hash,
            AccountToken.purpose.in_(
                [
                    "invite",
                    "password_change",
                ]
            ),
        )
    )

    if not record:
        raise HTTPException(
            status_code=400,
            detail="This activation link is invalid.",
        )

    if record.used_at is not None:
        raise HTTPException(
            status_code=409,
            detail="This activation link has already been used.",
        )

    now = datetime.now(timezone.utc)

    expires_at = _as_utc(record.expires_at)

    if expires_at <= now:
        raise HTTPException(
            status_code=410,
            detail="This activation link has expired.",
        )

    return record


@router.get("/validate")
def validate_invitation(
    token: str,
    db: Session = Depends(get_db),
):
    """
    Validate either:

    1. normal account invitation
    2. password-change invitation
    """

    record = get_invitation(
        db,
        token,
    )

    user = db.get(
        User,
        record.user_id,
    )

    if not user:
        raise HTTPException(
            status_code=400,
            detail="Invalid invitation.",
        )

    # ---------------------------------------------------------
    # Normal invitation
    # ---------------------------------------------------------

    if record.purpose == "invite":

        if (
            user.status
            != UserStatus.PENDING_INVITE.value
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "This invitation has "
                    "already been used."
                ),
            )

    # ---------------------------------------------------------
    # Password-change invitation
    # ---------------------------------------------------------

    elif record.purpose == "password_change":

        if (
            user.status
            != UserStatus.ACTIVE.value
        ):
            raise HTTPException(
                status_code=409,
                detail="This account is not active.",
            )

        if not user.must_change_password:
            raise HTTPException(
                status_code=409,
                detail=(
                    "This password-change link "
                    "is no longer required."
                ),
            )

    return {
        "valid": True,
        "name": user.name,
        "email": user.email,
        "purpose": record.purpose,
        "expires_at": _as_utc(
            record.expires_at
        ).isoformat(),
    }


@router.post("/complete", dependencies=[Depends(require_csrf)])
def complete_invitation(
    payload: CompleteInvitationRequest,
    db: Session = Depends(get_db),
):
    # ---------------------------------------------------------
    # 1. Password confirmation
    # ---------------------------------------------------------

    if payload.password != payload.confirm_password:
        raise HTTPException(
            status_code=400,
            detail="Passwords do not match.",
        )

    # ---------------------------------------------------------
    # 2. Password policy
    # ---------------------------------------------------------

    try:
        validate_password(
            payload.password
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    # ---------------------------------------------------------
    # 3. Resolve token
    # ---------------------------------------------------------

    record = get_invitation(
        db,
        payload.token,
    )

    # ---------------------------------------------------------
    # 4. Resolve user
    # ---------------------------------------------------------

    user = db.get(
        User,
        record.user_id,
    )

    if not user:
        raise HTTPException(
            status_code=400,
            detail="Invalid invitation.",
        )

    now = datetime.now(timezone.utc)

    # =========================================================
    # NORMAL INVITATION
    # =========================================================

    if record.purpose == "invite":

        if (
            user.status
            != UserStatus.PENDING_INVITE.value
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "This invitation has "
                    "already been used."
                ),
            )

        # Atomically claim token.
        claimed = db.execute(
            update(AccountToken)
            .where(
                AccountToken.id == record.id,
                AccountToken.used_at.is_(None),
            )
            .values(
                used_at=now,
            )
        )

        if claimed.rowcount != 1:
            db.rollback()

            raise HTTPException(
                status_code=409,
                detail=(
                    "This invitation has "
                    "already been used."
                ),
            )

        user.password_hash = hash_password(
            payload.password
        )

        user.status = (
            UserStatus.ACTIVE.value
        )

        user.email_verified = True

        user.must_change_password = False

        # Invalidate other invite tokens.
        db.execute(
            update(AccountToken)
            .where(
                AccountToken.user_id == user.id,
                AccountToken.purpose == "invite",
                AccountToken.id != record.id,
                AccountToken.used_at.is_(None),
            )
            .values(
                used_at=now,
            )
        )

        db.commit()

        return {
            "success": True,
            "purpose": "invite",
            "message": (
                "Account activated successfully."
            ),
        }

    # =========================================================
    # PASSWORD CHANGE
    # =========================================================

    if record.purpose == "password_change":

        if (
            user.status
            != UserStatus.ACTIVE.value
        ):
            raise HTTPException(
                status_code=409,
                detail="This account is not active.",
            )

        if not user.must_change_password:
            raise HTTPException(
                status_code=409,
                detail=(
                    "This password-change link "
                    "has already been completed."
                ),
            )

        # Atomically claim token.
        claimed = db.execute(
            update(AccountToken)
            .where(
                AccountToken.id == record.id,
                AccountToken.used_at.is_(None),
            )
            .values(
                used_at=now,
            )
        )

        if claimed.rowcount != 1:
            db.rollback()

            raise HTTPException(
                status_code=409,
                detail=(
                    "This password-change link "
                    "has already been used."
                ),
            )

        # Replace temporary password.
        user.password_hash = hash_password(
            payload.password
        )

        user.must_change_password = False

        user.email_verified = True

        # Invalidate any other password-change
        # tokens belonging to this user.
        db.execute(
            update(AccountToken)
            .where(
                AccountToken.user_id == user.id,
                AccountToken.purpose
                == "password_change",
                AccountToken.id != record.id,
                AccountToken.used_at.is_(None),
            )
            .values(
                used_at=now,
            )
        )

        db.commit()

        return {
            "success": True,
            "purpose": "password_change",
            "message": (
                "Password changed successfully. "
                "You can now sign in to AMCShield "
                "using your new password."
            ),
        }

    raise HTTPException(
        status_code=400,
        detail="Unsupported account token.",
    )