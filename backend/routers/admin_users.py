from datetime import datetime, timedelta, timezone

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)

from pydantic import (
    BaseModel,
    EmailStr,
)

from sqlalchemy import select

from sqlalchemy.orm import Session

from ..config import settings

from ..database.db import get_db

from ..database.models import (
    AccountToken,
    AuditEvent,
    User,
    UserRole,
    UserStatus,
)

from ..services.email_service import (
    send_invitation_email,
)

from ..services.security_service import (
    generate_token,
    hash_token,
)


router = APIRouter(
    prefix="/admin/users",
    tags=["Admin User Management"],
)


class InviteUserRequest(BaseModel):

    name: str

    email: EmailStr


def require_admin(
    request: Request,
):

    session_user = request.session.get(
        "user"
    )

    if not session_user:

        raise HTTPException(
            status_code=401,
            detail="Not authenticated.",
        )

    role = session_user.get(
        "role"
    )

    if role not in (
        "Administrator",
        "admin",
    ):

        raise HTTPException(
            status_code=403,
            detail="Administrator access required.",
        )

    return session_user


@router.post("/invite")
def invite_user(

    payload: InviteUserRequest,

    request: Request,

    db: Session = Depends(get_db),

):

    admin_user = require_admin(
        request
    )

    name = payload.name.strip()

    email = (
        str(payload.email)
        .strip()
        .lower()
    )

    if not name:

        raise HTTPException(
            status_code=400,
            detail="Name is required.",
        )


    # ---------------------------------------------------------
    # Prevent duplicate accounts
    # ---------------------------------------------------------

    existing_user = db.scalar(

        select(User).where(
            User.email == email
        )

    )

    if existing_user:

        raise HTTPException(
            status_code=409,
            detail=(
                "An AMCShield account already "
                "exists for this email."
            ),
        )


    # ---------------------------------------------------------
    # Create pending user
    # ---------------------------------------------------------

    user = User(

        name=name,

        email=email,

        role=UserRole.USER.value,

        status=UserStatus.PENDING_INVITE.value,

        must_change_password=True,

        email_verified=False,

    )

    db.add(user)

    db.flush()


    # ---------------------------------------------------------
    # Generate single-use activation token
    # ---------------------------------------------------------

    raw_token = generate_token()

    token_hash = hash_token(
        raw_token
    )


    token = AccountToken(

        user_id=user.id,

        token_hash=token_hash,

        purpose="invite",

        expires_at=(
            datetime.now(timezone.utc)
            + timedelta(
                minutes=settings.invitation_minutes
            )
        ),

    )

    db.add(token)


    # ---------------------------------------------------------
    # Audit
    # ---------------------------------------------------------

    actor_id = admin_user.get(
        "id"
    )

    if not isinstance(
        actor_id,
        int,
    ):

        actor_id = None


    audit = AuditEvent(

        actor_user_id=actor_id,

        action="user.invited",

        target_user_id=user.id,

        ip_address=(
            request.client.host
            if request.client
            else None
        ),

    )

    db.add(audit)


    db.commit()


    # ---------------------------------------------------------
    # Create activation URL
    # ---------------------------------------------------------

    invitation_url = (

        f"{settings.app_base_url.rstrip('/')}"
        f"/invite.html?token={raw_token}"

    )


    # ---------------------------------------------------------
    # Send email
    # ---------------------------------------------------------

    try:

        send_invitation_email(

            name=name,

            email=email,

            invitation_url=invitation_url,

        )

    except Exception as exc:

        print(
            f"Invitation email error: {exc}"
        )

        # IMPORTANT:
        # User exists but invitation wasn't delivered.
        # Admin can resend later.

        raise HTTPException(

            status_code=502,

            detail=(
                "Account was created, but the "
                "invitation email could not be sent. "
                "Check SMTP configuration."
            ),

        )


    return {

        "success": True,

        "user_id": user.id,

        "status": user.status,

        "message": (
            "Account created and activation "
            "email sent successfully."
        ),

    }