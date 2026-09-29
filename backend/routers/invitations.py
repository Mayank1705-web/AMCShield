from datetime import datetime, timezone

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from pydantic import BaseModel, Field

from sqlalchemy import select

from sqlalchemy.orm import Session

from ..database.db import get_db

from ..database.models import (
    AccountToken,
    User,
    UserStatus,
)

from ..services.security_service import (
    hash_password,
    hash_token,
    validate_password,
)


router = APIRouter(
    prefix="/account/invite",
    tags=["Account Invitation"],
)


class CompleteInvitationRequest(BaseModel):

    token: str = Field(
        min_length=20
    )

    password: str = Field(
        min_length=12,
        max_length=128,
    )

    confirm_password: str = Field(
        min_length=12,
        max_length=128,
    )


def get_invitation(
    db: Session,
    raw_token: str,
):

    record = db.scalar(

        select(AccountToken).where(

            AccountToken.token_hash
            == hash_token(raw_token),

            AccountToken.purpose
            == "invite",

            AccountToken.used_at.is_(None),

        )

    )


    if not record:

        raise HTTPException(
            status_code=400,
            detail=(
                "This activation link "
                "is invalid or expired."
            ),
        )


    now = datetime.now(
        timezone.utc
    )


    if record.expires_at <= now:

        raise HTTPException(
            status_code=400,
            detail=(
                "This activation link "
                "is invalid or expired."
            ),
        )


    return record


@router.get("/validate")
def validate_invitation(

    token: str,

    db: Session = Depends(get_db),

):

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


    if (
        user.status
        != UserStatus.PENDING_INVITE.value
    ):

        raise HTTPException(
            status_code=400,
            detail="This invitation has already been used.",
        )


    return {

        "valid": True,

        "name": user.name,

        "email": user.email,

        "expires_at":
            record.expires_at.isoformat(),

    }


@router.post("/complete")
def complete_invitation(

    payload:
        CompleteInvitationRequest,

    db: Session = Depends(get_db),

):

    if (
        payload.password
        != payload.confirm_password
    ):

        raise HTTPException(
            status_code=400,
            detail="Passwords do not match.",
        )


    try:

        validate_password(
            payload.password
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


    record = get_invitation(
        db,
        payload.token,
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


    user.password_hash = hash_password(
        payload.password
    )

    user.status = (
        UserStatus.ACTIVE.value
    )

    user.email_verified = True

    user.must_change_password = False


    record.used_at = datetime.now(
        timezone.utc
    )


    db.commit()


    return {

        "success": True,

        "message":
            "Account activated successfully.",

    }