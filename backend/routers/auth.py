from __future__ import annotations

import secrets
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
from datetime import datetime, timezone
from pathlib import Path
import json

from pydantic import BaseModel, EmailStr

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


@router.post("/contact-admin")
def contact_admin(request: ContactAdminRequest):
    messages_file = settings.results_dir / "admin_messages.json"

    messages_file.parent.mkdir(parents=True, exist_ok=True)

    messages = []

    if messages_file.exists():
        try:
            with messages_file.open("r", encoding="utf-8") as f:
                messages = json.load(f)

            if not isinstance(messages, list):
                messages = []

        except (json.JSONDecodeError, OSError):
            messages = []

    message_record = {
        "id": len(messages) + 1,
        "name": request.name.strip(),
        "email": str(request.email),
        "subject": request.subject.strip(),
        "message": request.message.strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "new",
    }

    messages.append(message_record)

    with messages_file.open("w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2)

    return {
        "success": True,
        "message": "Your message has been submitted to the AMCShield administrator.",
        "message_id": message_record["id"],
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
async def github_callback(request: Request, code: str | None = None, state: str | None = None):
    """Exchange the GitHub authorization code and create the AMCShield session."""
    expected_state = request.session.pop("github_oauth_state", None)
    if not code or not state or not expected_state or not secrets.compare_digest(state, expected_state):
        return RedirectResponse(url="/?auth_error=invalid_oauth_state", status_code=302)

    if not settings.github_client_id or not settings.github_client_secret:
        return RedirectResponse(url="/?auth_error=github_not_configured", status_code=302)

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
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
                raise RuntimeError(token_data.get("error_description") or "GitHub did not return an access token")

            headers = {
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {access_token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "AMCShield",
            }
            user_response = await client.get(GITHUB_USER_URL, headers=headers)
            user_response.raise_for_status()
            github_user = user_response.json()

            email = github_user.get("email")
            if not email:
                email_response = await client.get(GITHUB_EMAILS_URL, headers=headers)
                if email_response.is_success:
                    emails = email_response.json()
                    primary = next((e for e in emails if e.get("primary")), None)
                    email = (primary or (emails[0] if emails else {})).get("email")

        request.session["user"] = {
            "provider": "github",
            "id": github_user.get("id"),
            "login": github_user.get("login"),
            "name": github_user.get("name") or github_user.get("login") or "GitHub User",
            "email": email,
            "avatar_url": github_user.get("avatar_url"),
            "role": "GitHub User",
        }
        return RedirectResponse(url="/dashboard.html", status_code=302)

    except Exception as exc:
        # Do not expose provider secrets or raw upstream responses to the browser.
        print(f"GitHub OAuth error: {exc}")
        return RedirectResponse(url="/?auth_error=github_auth_failed", status_code=302)


@router.post("/login")
async def login(payload: LoginRequest, request: Request):
    """Local development login using credentials stored in .env."""
    if not settings.admin_password:
        raise HTTPException(
            status_code=503,
            detail="Local login is not configured. Use GitHub login or set ADMIN_PASSWORD in .env.",
        )

    identity_ok = secrets.compare_digest(payload.identity.strip().lower(), settings.admin_username.strip().lower()) or secrets.compare_digest(payload.identity.strip().lower(), settings.admin_email.strip().lower())
    password_ok = secrets.compare_digest(payload.password, settings.admin_password)

    if not identity_ok or not password_ok:
        raise HTTPException(status_code=401, detail="Invalid username/email or password.")

    request.session["user"] = {
        "provider": "local",
        "id": "local-admin",
        "login": settings.admin_username,
        "name": settings.admin_username,
        "email": settings.admin_email,
        "avatar_url": None,
        "role": "Administrator",
    }
    return {"success": True, "user": request.session["user"]}


@router.get("/me")
async def me(request: Request):
    user = _session_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"authenticated": True, "user": user}


@router.post("/logout")
async def logout(request: Request):
    request.session.clear()
    return {"success": True}
