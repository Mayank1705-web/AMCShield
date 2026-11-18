from __future__ import annotations

import secrets
from urllib.parse import urlencode
from time import time

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse

from pydantic import BaseModel, EmailStr

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database.db import get_db
from ..database.models import (
    AccessRequest,
    AccessRequestStatus,
    User,
    UserRole,
    UserStatus,
    utcnow,
)
from ..services.security_service import (
    check_login_rate_limit,
    clear_login_failures,
    get_csrf_token,
    hash_password,
    require_csrf,
    record_login_failure,
    verify_password,
)

from ..config import settings

router = APIRouter(prefix="/auth", tags=["Authentication"])

GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"
GITHUB_EMAILS_URL = "https://api.github.com/user/emails"


class LoginRequest(BaseModel):
    identity: str
    password: str


class ContactAdminRequest(BaseModel):
    name: str
    email: EmailStr
    subject: str
    message: str


@router.post("/contact-admin", dependencies=[Depends(require_csrf)])
def contact_admin(
    payload: ContactAdminRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    name = payload.name.strip()
    email = str(payload.email).strip().lower()
    subject = payload.subject.strip()
    message = payload.message.strip()

    if not name or len(name) > 120:
        raise HTTPException(
            status_code=422,
            detail="Name must contain 1 to 120 characters.",
        )

    if not subject or len(subject) > 200:
        raise HTTPException(
            status_code=422,
            detail="Subject must contain 1 to 200 characters.",
        )

    if not message or len(message) > 5000:
        raise HTTPException(
            status_code=422,
            detail="Message must contain 1 to 5000 characters.",
        )

    # Avoid duplicate active requests from the same email.
    existing = db.scalar(
        select(AccessRequest).where(
            AccessRequest.email == email,
            AccessRequest.status.in_(
                [
                    AccessRequestStatus.NEW.value,
                    AccessRequestStatus.READ.value,
                ]
            ),
        )
    )

    if existing:
        return {
            "success": True,
            "message": (
                "Your message has been submitted "
                "to the AMCShield administrator."
            ),
            "message_id": existing.id,
        }

    client_ip = (
        request.client.host
        if request.client
        else None
    )

    access_request = AccessRequest(
        name=name,
        email=email,
        subject=subject,
        message=message,
        status=AccessRequestStatus.NEW.value,
        ip_address=client_ip,
    )

    db.add(access_request)
    db.commit()
    db.refresh(access_request)

    return {
        "success": True,
        "message": (
            "Your message has been submitted "
            "to the AMCShield administrator."
        ),
        "message_id": access_request.id,
    }


def _frontend_url() -> str:
    return "/"


def _session_user(request: Request):
    return request.session.get("user")


@router.get("/github/login")
async def github_login(request: Request):
    """Start GitHub OAuth authorization."""
    if not settings.github_client_id or not settings.github_client_secret:
        return RedirectResponse(
            url="/?auth_error=github_not_configured",
            status_code=302,
        )

    state = secrets.token_urlsafe(32)
    request.session["github_oauth_state"] = state
    request.session["github_oauth_state_created_at"] = int(time())

    params = {
        "client_id": settings.github_client_id,
        "redirect_uri": settings.github_redirect_uri,
        "scope": settings.github_scope,
        "state": state,
        "allow_signup": "true",
    }
    return RedirectResponse(
        url=f"{GITHUB_AUTHORIZE_URL}?{urlencode(params)}",
        status_code=302,
    )


@router.get("/github/callback")
async def github_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    db: Session = Depends(get_db),
):
    """Complete GitHub OAuth only for an existing active AMCShield account."""
    expected_state = request.session.pop("github_oauth_state", None)
    state_created_at = request.session.pop(
        "github_oauth_state_created_at",
        None,
    )

    state_is_fresh = False
    if isinstance(state_created_at, int):
        state_is_fresh = (
            0 <= time() - state_created_at
            <= settings.github_state_max_age
        )

    if (
        not code
        or not state
        or not expected_state
        or not state_is_fresh
        or not secrets.compare_digest(state, expected_state)
    ):
        return RedirectResponse(
            url="/?auth_error=invalid_oauth_state",
            status_code=302,
        )

    if not settings.github_client_id or not settings.github_client_secret:
        return RedirectResponse(
            url="/?auth_error=github_not_configured",
            status_code=302,
        )

    try:
        async with httpx.AsyncClient(
            timeout=15.0,
            follow_redirects=False,
        ) as client:
            token_response = await client.post(
                GITHUB_TOKEN_URL,
                data={
                    "client_id": settings.github_client_id,
                    "client_secret": settings.github_client_secret,
                    "code": code,
                    "redirect_uri": settings.github_redirect_uri,
                },
                headers={
                    "Accept": "application/json",
                    "User-Agent": "AMCShield",
                },
            )
            token_response.raise_for_status()
            token_data = token_response.json()
            access_token = token_data.get("access_token")
            if not access_token:
                raise RuntimeError("GitHub did not return an access token.")

            github_headers = {
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {access_token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "AMCShield",
            }

            user_response = await client.get(
                GITHUB_USER_URL,
                headers=github_headers,
            )
            user_response.raise_for_status()
            github_user = user_response.json()

            # Always consult GitHub's email endpoint and require a verified
            # email. The public profile email alone is not sufficient.
            email_response = await client.get(
                GITHUB_EMAILS_URL,
                headers=github_headers,
            )
            email_response.raise_for_status()
            emails = email_response.json()

            verified_emails = [
                item
                for item in emails
                if item.get("verified") and item.get("email")
            ]
            primary_verified = next(
                (
                    item
                    for item in verified_emails
                    if item.get("primary")
                ),
                None,
            )
            selected_email = (
                primary_verified or
                (verified_emails[0] if verified_emails else None)
            )

        if not selected_email:
            return RedirectResponse(
                url="/?auth_error=github_email_unverified",
                status_code=302,
            )

        email = str(selected_email["email"]).strip().lower()
        db_user = db.scalar(
            select(User).where(User.email == email)
        )

        # GitHub authentication does not itself authorize AMCShield access.
        if not db_user:
            return RedirectResponse(
                url="/?auth_error=github_account_not_authorized",
                status_code=302,
            )

        if db_user.status != UserStatus.ACTIVE.value:
            return RedirectResponse(
                url="/?auth_error=github_account_not_active",
                status_code=302,
            )

        db_user.last_login_at = utcnow()
        db.commit()

        session_user = {
            "provider": "github",
            "id": db_user.id,
            "github_id": github_user.get("id"),
            "login": github_user.get("login"),
            "name": db_user.name,
            "email": db_user.email,
            "avatar_url": github_user.get("avatar_url"),
            # These values are derived from the DB record, not GitHub/client input.
            "role": (
                "Administrator"
                if db_user.role == UserRole.ADMIN.value
                else "User"
            ),
            "role_key": db_user.role,
        }

        # Replace any pre-authentication session state to prevent fixation.
        _start_session(request, session_user)

        return RedirectResponse(
            url="/dashboard.html",
            status_code=302,
        )

    except Exception as exc:
        # Do not expose provider secrets or raw upstream responses.
        print(f"GitHub OAuth error: {type(exc).__name__}")
        return RedirectResponse(
            url="/?auth_error=github_auth_failed",
            status_code=302,
        )


_DUMMY_HASH = hash_password("Timing-Dummy-Password-1!")

_ROLE_LABELS = {
    UserRole.ADMIN.value: "Administrator",
    UserRole.USER.value: "User",
}


def _bootstrap_admin_matches(identity: str, password: str) -> bool:
    if not settings.admin_password:
        return False

    identity_ok = identity in (
        settings.admin_username.strip().lower(),
        settings.admin_email.strip().lower(),
    )

    password_ok = secrets.compare_digest(
        password.encode("utf-8"),
        settings.admin_password.encode("utf-8"),
    )

    return identity_ok and password_ok


def _start_session(request: Request, session_user: dict) -> None:
    request.session.clear()
    request.session["user"] = session_user


@router.get("/csrf")
def csrf(request: Request):
    """Issue the per-session CSRF token used by state-changing API calls."""
    return {"csrf_token": get_csrf_token(request)}


@router.post("/login")
def login(
    payload: LoginRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Login for:
    1. Bootstrap administrator from .env
    2. Activated database users
    """

    identity = payload.identity.strip().lower()
    client_ip = request.client.host if request.client else None

    # Check the same rate-limit bucket before doing password work.
    retry_after = check_login_rate_limit(db, identity, client_ip)
    if retry_after:
        raise HTTPException(
            status_code=429,
            detail="Too many failed login attempts. Please try again later.",
            headers={"Retry-After": str(retry_after)},
        )

    # ---------------------------------------------------------
    # 1. Bootstrap administrator
    # ---------------------------------------------------------

    if _bootstrap_admin_matches(identity, payload.password):
        clear_login_failures(db, identity, client_ip)

        session_user = {
            "provider": "local",
            "id": "local-admin",
            "login": settings.admin_username,
            "name": settings.admin_username,
            "email": settings.admin_email,
            "avatar_url": None,
            "role": "Administrator",
            "role_key": "admin",
        }

        _start_session(request, session_user)

        return {
            "success": True,
            "user": session_user,
        }

    # ---------------------------------------------------------
    # 2. Database-backed invited user
    # ---------------------------------------------------------

    user = db.scalar(
        select(User).where(User.email == identity)
    )

    # Do a dummy Argon2 verification so nonexistent-user
    # responses don't have an obvious timing difference.

    if user is None:
        verify_password(
            _DUMMY_HASH,
            payload.password,
        )
        retry_after = record_login_failure(db, identity, client_ip)
        headers = {"Retry-After": str(retry_after)} if retry_after else None
        raise HTTPException(
            status_code=429 if retry_after else 401,
            detail=(
                "Too many failed login attempts. Please try again later."
                if retry_after
                else "Invalid username/email or password."
            ),
            headers=headers,
        )

    # ---------------------------------------------------------
    # 3. Pending invitation
    # ---------------------------------------------------------

    if user.status != UserStatus.ACTIVE.value:
        retry_after = record_login_failure(db, identity, client_ip)
        headers = {"Retry-After": str(retry_after)} if retry_after else None
        raise HTTPException(
            status_code=429 if retry_after else 401,
            detail=(
                "Too many failed login attempts. Please try again later."
                if retry_after
                else "Invalid username/email or password."
            ),
            headers=headers,
        )

    # ---------------------------------------------------------
    # 5. Verify Argon2 password
    # ---------------------------------------------------------

    if (
        not user.password_hash
        or not verify_password(
            user.password_hash,
            payload.password,
        )
    ):
        retry_after = record_login_failure(db, identity, client_ip)
        headers = {"Retry-After": str(retry_after)} if retry_after else None
        raise HTTPException(
            status_code=429 if retry_after else 401,
            detail=(
                "Too many failed login attempts. Please try again later."
                if retry_after
                else "Invalid username/email or password."
            ),
            headers=headers,
        )

    # ---------------------------------------------------------
    # 6. Update last login
    # ---------------------------------------------------------

    user.last_login_at = utcnow()
    db.commit()
    clear_login_failures(db, identity, client_ip)

    # ---------------------------------------------------------
    # 7. Create session
    # ---------------------------------------------------------

    session_user = {
        "provider": "local",
        "id": user.id,
        "login": user.email,
        "name": user.name,
        "email": user.email,
        "avatar_url": None,
        "role": _ROLE_LABELS.get(
            user.role,
            "User",
        ),
        "role_key": user.role,
    }

    _start_session(
        request,
        session_user,
    )

    return {
        "success": True,
        "user": session_user,
    }


@router.get("/me")
async def me(request: Request):
    user = _session_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"authenticated": True, "user": user}


@router.post("/logout", dependencies=[Depends(require_csrf)])
async def logout(request: Request):
    request.session.clear()
    return {"success": True}