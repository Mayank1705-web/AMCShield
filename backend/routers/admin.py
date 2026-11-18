from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..dependencies import require_admin
from ..database.db import get_db
from ..database.models import (
    AccessRequest,
    AccessRequestStatus,
    AccountToken,
    AuditEvent,
    User,
    UserRole,
    UserStatus,
    utcnow,
)
from ..services.email_service import (
    send_credentials_email,
)
from ..services.security_service import (
    generate_token,
    hash_token,
    require_csrf,
)


router = APIRouter(
    prefix="/admin",
    tags=["Admin"],
)


class CloseRequest(BaseModel):
    admin_note: str | None = Field(
        default=None,
        max_length=5000,
    )


def _actor_id(
    user: dict[str, Any],
) -> int | None:
    value = user.get("id")

    if isinstance(value, int):
        return value

    return None


def _request_dict(
    item: AccessRequest,
) -> dict[str, Any]:
    return {
        "id": item.id,
        "name": item.name,
        "email": item.email,
        "subject": item.subject,
        "message": item.message,
        "status": item.status,
        "created_at": item.created_at.isoformat()
        if item.created_at
        else None,
        "reviewed_at": item.reviewed_at.isoformat()
        if item.reviewed_at
        else None,
        "reviewed_by": item.reviewed_by,
        "admin_note": item.admin_note,
        "user_id": item.user_id,
        "ip_address": item.ip_address,
    }


def _credential_link_token(
    db: Session,
    user_id: int,
) -> str:
    raw_token = generate_token()

    token = AccountToken(
        user_id=user_id,
        token_hash=hash_token(raw_token),
        purpose="password_change",
        expires_at=utcnow() + timedelta(
            hours=24
        ),
    )

    db.add(token)

    return raw_token


@router.get("/messages")
def list_messages(
    _: dict[str, Any] = Depends(
        require_admin
    ),
    db: Session = Depends(get_db),
):
    messages = list(
        db.scalars(
            select(AccessRequest).order_by(
                AccessRequest.created_at.desc()
            )
        )
    )

    counts = {
        "total": len(messages),
        "new": sum(
            1
            for item in messages
            if item.status
            == AccessRequestStatus.NEW.value
        ),
        "read": sum(
            1
            for item in messages
            if item.status
            == AccessRequestStatus.READ.value
        ),
        "approved": sum(
            1
            for item in messages
            if item.status
            == AccessRequestStatus.APPROVED.value
        ),
        "closed": sum(
            1
            for item in messages
            if item.status
            == AccessRequestStatus.CLOSED.value
        ),
    }

    return {
        "success": True,
        "messages": [
            _request_dict(item)
            for item in messages
        ],
        "counts": counts,
    }


@router.get("/messages/{message_id}")
def get_message(
    message_id: int,
    _: dict[str, Any] = Depends(
        require_admin
    ),
    db: Session = Depends(get_db),
):
    message = db.get(
        AccessRequest,
        message_id,
    )

    if not message:
        raise HTTPException(
            status_code=404,
            detail="Message not found.",
        )

    # Automatically mark new → read.
    if (
        message.status
        == AccessRequestStatus.NEW.value
    ):
        message.status = (
            AccessRequestStatus.READ.value
        )
        db.commit()
        db.refresh(message)

    return {
        "success": True,
        "message": _request_dict(
            message
        ),
    }


@router.post(
    "/messages/{message_id}/approve",
    dependencies=[Depends(require_csrf)],
)
def approve_message(
    message_id: int,
    request: Request,
    admin_user: dict[str, Any] = Depends(
        require_admin
    ),
    db: Session = Depends(get_db),
):
    access_request = db.get(
        AccessRequest,
        message_id,
    )

    if not access_request:
        raise HTTPException(
            status_code=404,
            detail="Message not found.",
        )

    if access_request.status in {
        AccessRequestStatus.APPROVED.value,
        AccessRequestStatus.CLOSED.value,
    }:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Request is already "
                f"{access_request.status}."
            ),
        )

    email = (
        access_request.email
        .strip()
        .lower()
    )

    existing_user = db.scalar(
        select(User).where(
            User.email == email
        )
    )

    if existing_user:
        raise HTTPException(
            status_code=409,
            detail="Account already exists.",
        )

    # Do not generate or email a temporary password.
    # The user creates their own password through the one-time setup link.
    user = User(
        name=access_request.name,
        email=email,
        role=UserRole.USER.value,
        status=UserStatus.ACTIVE.value,
        must_change_password=True,
        email_verified=True,
        password_hash=None,
    )

    db.add(user)
    db.flush()

    raw_token = _credential_link_token(
        db,
        user.id,
    )

    now = utcnow()

    access_request.status = (
        AccessRequestStatus.APPROVED.value
    )
    access_request.reviewed_at = now
    access_request.reviewed_by = (
        _actor_id(admin_user)
    )
    access_request.user_id = user.id

    db.add(
        AuditEvent(
            actor_user_id=_actor_id(
                admin_user
            ),
            action="request.approved",
            target_user_id=user.id,
            ip_address=(
                request.client.host
                if request.client
                else None
            ),
            metadata_json=json.dumps(
                {
                    "request_id": access_request.id,
                    "email": email,
                }
            ),
        )
    )

    db.add(
        AuditEvent(
            actor_user_id=_actor_id(
                admin_user
            ),
            action="user.created",
            target_user_id=user.id,
            ip_address=(
                request.client.host
                if request.client
                else None
            ),
            metadata_json=json.dumps(
                {
                    "request_id": access_request.id,
                    "email": email,
                }
            ),
        )
    )

    login_url = (
        f"{settings.app_base_url}/"
    )

    change_password_url = (
        f"{settings.app_base_url}"
        f"/invite.html?token={raw_token}"
    )

    try:
        send_credentials_email(
            name=user.name,
            email=user.email,
            login_url=login_url,
            change_password_url=change_password_url,
            expires_hours=24,
        )

    except Exception as exc:
        db.rollback()

        raise HTTPException(
            status_code=502,
            detail=(
                "Email could not be sent — "
                "no account was created. "
                "Check SMTP settings."
            ),
        ) from exc

    db.commit()

    return {
        "success": True,
        "user_id": user.id,
        "email": user.email,
        "status": (
            AccessRequestStatus.APPROVED.value
        ),
        "message": (
            f"Password setup link emailed to "
            f"{user.email}"
        ),
    }


@router.post(
    "/messages/{message_id}/close",
    dependencies=[Depends(require_csrf)],
)
def close_message(
    message_id: int,
    payload: CloseRequest,
    request: Request,
    admin_user: dict[str, Any] = Depends(
        require_admin
    ),
    db: Session = Depends(get_db),
):
    access_request = db.get(
        AccessRequest,
        message_id,
    )

    if not access_request:
        raise HTTPException(
            status_code=404,
            detail="Message not found.",
        )

    if access_request.status in {
        AccessRequestStatus.APPROVED.value,
        AccessRequestStatus.CLOSED.value,
    }:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Request is already "
                f"{access_request.status}."
            ),
        )

    access_request.status = (
        AccessRequestStatus.CLOSED.value
    )

    access_request.reviewed_at = utcnow()

    access_request.reviewed_by = (
        _actor_id(admin_user)
    )

    access_request.admin_note = (
        payload.admin_note.strip()
        if payload.admin_note
        else None
    )

    db.add(
        AuditEvent(
            actor_user_id=_actor_id(
                admin_user
            ),
            action="request.closed",
            target_user_id=None,
            ip_address=(
                request.client.host
                if request.client
                else None
            ),
            metadata_json=json.dumps(
                {
                    "request_id": access_request.id,
                }
            ),
        )
    )

    db.commit()
    db.refresh(access_request)

    return {
        "success": True,
        "status": access_request.status,
        "message": (
            "Access request closed."
        ),
    }


@router.post(
    "/messages/{message_id}/resend",
    dependencies=[Depends(require_csrf)],
)
def resend_credentials(
    message_id: int,
    request: Request,
    admin_user: dict[str, Any] = Depends(
        require_admin
    ),
    db: Session = Depends(get_db),
):
    access_request = db.get(
        AccessRequest,
        message_id,
    )

    if not access_request:
        raise HTTPException(
            status_code=404,
            detail="Message not found.",
        )

    if (
        access_request.status
        != AccessRequestStatus.APPROVED.value
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "Credentials can only be "
                "resent for approved requests."
            ),
        )

    if not access_request.user_id:
        raise HTTPException(
            status_code=409,
            detail=(
                "Approved request has no "
                "associated user account."
            ),
        )

    user = db.get(
        User,
        access_request.user_id,
    )

    if not user:
        raise HTTPException(
            status_code=409,
            detail=(
                "Associated user account "
                "does not exist."
            ),
        )

    # Resend a one-time password setup link without generating
    # or transmitting a temporary password.
    user.must_change_password = True

    # Invalidate previous password-change tokens.
    now = utcnow()

    db.query(AccountToken).filter(
        AccountToken.user_id == user.id,
        AccountToken.purpose
        == "password_change",
        AccountToken.used_at.is_(None),
    ).update(
        {
            AccountToken.used_at: now,
        },
        synchronize_session=False,
    )

    raw_token = _credential_link_token(
        db,
        user.id,
    )

    login_url = (
        f"{settings.app_base_url}/"
    )

    change_password_url = (
        f"{settings.app_base_url}"
        f"/invite.html?token={raw_token}"
    )

    try:
        send_credentials_email(
            name=user.name,
            email=user.email,
            login_url=login_url,
            change_password_url=change_password_url,
            expires_hours=24,
        )

    except Exception as exc:
        db.rollback()

        raise HTTPException(
            status_code=502,
            detail=(
                "Email could not be sent."
            ),
        ) from exc

    db.commit()

    return {
        "success": True,
        "email": user.email,
        "message": (
            f"Password setup link re-sent to "
            f"{user.email}"
        ),
    }