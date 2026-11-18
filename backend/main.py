from pathlib import Path
import hmac
import mimetypes

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy.orm import Session

from .database.db import get_db, init_db
from .config import settings
from .dependencies import is_admin_session
from .routers import (
    admin,
    admin_users,
    attacks,
    dashboard,
    dataset,
    evaluation,
    invitations,
    project,
    reports,
    training,
    visualization,
)
from .routers.auth import router as auth_router


# ============================================================
# MIME TYPES
# ============================================================

mimetypes.add_type("text/css", ".css", strict=True)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="FastAPI backend for the AMCShield research platform.",
    docs_url="/docs" if settings.environment.strip().lower() != "production" else None,
    redoc_url="/redoc" if settings.environment.strip().lower() != "production" else None,
    openapi_url="/openapi.json" if settings.environment.strip().lower() != "production" else None,
)


# ============================================================
# DATABASE
# ============================================================

init_db()


# ============================================================
# SESSION MIDDLEWARE
# ============================================================
#
# SessionMiddleware must be available before anything that
# accesses request.session.
#
# We register it first because FastAPI/Starlette wraps later
# middleware around earlier middleware.
#
# ============================================================

app.add_middleware(
    SessionMiddleware,
    secret_key=settings.session_secret,
    session_cookie="amcshield_session",
    max_age=settings.session_max_age,
    same_site="lax",
    https_only=settings.session_https_only,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=[
        "GET",
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
        "OPTIONS",
    ],
    allow_headers=[
        "Content-Type",
        "Accept",
        "X-CSRF-Token",
    ],
)


# ============================================================
# SECURITY MIDDLEWARE
# ============================================================
#
# IMPORTANT:
#
# This middleware does NOT access request.session.
#
# CSRF is handled by the dedicated /api/auth/csrf endpoint
# and route-level validation in the authentication/security
# layer.
#
# Keeping session-dependent CSRF logic out of this middleware
# eliminates the SessionMiddleware ordering problem permanently.
#
# ============================================================

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    response = await call_next(request)

    # --------------------------------------------------------
    # Browser security headers
    # --------------------------------------------------------

    response.headers["X-Content-Type-Options"] = "nosniff"

    response.headers["X-Frame-Options"] = "DENY"

    response.headers["Referrer-Policy"] = (
        "strict-origin-when-cross-origin"
    )

    response.headers["Permissions-Policy"] = (
        "camera=(), "
        "microphone=(), "
        "geolocation=()"
    )

    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com data:; "
        "img-src 'self' data: blob:; "
        "connect-src 'self'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "form-action 'self'; "
        "frame-ancestors 'none'; "
        "manifest-src 'self'; "
        "worker-src 'self' blob:;"
    )

    # --------------------------------------------------------
    # HSTS
    # --------------------------------------------------------

    if settings.session_https_only:
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )

    # --------------------------------------------------------
    # API cache control
    # --------------------------------------------------------

    if request.url.path.startswith(
        f"{settings.api_prefix}/"
    ) or request.url.path.endswith(".html") or request.url.path in {"/", "/admin"}:
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"

    return response


# ============================================================
# API ROUTERS
# ============================================================

for router in (
    auth_router,
    admin.router,
    admin_users.router,
    dashboard.router,
    dataset.router,
    training.router,
    attacks.router,
    evaluation.router,
    visualization.router,
    reports.router,
    project.router,
    invitations.router,
):
    app.include_router(
        router,
        prefix=settings.api_prefix,
    )


# ============================================================
# HEALTH ENDPOINTS
# ============================================================

@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
    }


@app.get("/api/health", tags=["Health"])
def api_health():
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
    }


# ============================================================
# FRONTEND
# ============================================================

FRONTEND_DIR = settings.frontend_dir.resolve()

if not FRONTEND_DIR.exists():
    raise RuntimeError(
        f"AMCShield frontend directory not found: {FRONTEND_DIR}"
    )

if not (FRONTEND_DIR / "index.html").is_file():
    raise RuntimeError(
        "AMCShield frontend index.html not found: "
        f"{FRONTEND_DIR / 'index.html'}"
    )


NO_STORE = {"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"}


def _register_admin_asset(
    url_path: str,
    filename: str,
    media_type: str | None,
    is_page: bool,
):
    @app.get(url_path, include_in_schema=False)
    def _serve(
        request: Request,
        db: Session = Depends(get_db),
    ):
        if is_admin_session(request, db):
            return FileResponse(
                FRONTEND_DIR / filename,
                media_type=media_type,
                headers=NO_STORE,
            )

        if not is_page:
            return Response(
                status_code=404,
                headers=NO_STORE,
            )

        if request.session.get("user"):
            return RedirectResponse(
                "/dashboard.html",
                status_code=302,
                headers=NO_STORE,
            )

        return RedirectResponse(
            "/",
            status_code=302,
            headers=NO_STORE,
        )


_register_admin_asset("/admin.html", "admin.html", "text/html", True)
_register_admin_asset("/admin", "admin.html", "text/html", True)
_register_admin_asset("/admin.js", "admin.js", "application/javascript", False)
_register_admin_asset("/admin.css", "admin.css", "text/css", False)


app.mount(
    "/",
    StaticFiles(
        directory=str(FRONTEND_DIR),
        html=True,
    ),
    name="frontend",
)


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )