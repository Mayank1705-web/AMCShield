from pathlib import Path
import mimetypes
from .database.db import init_db
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from .config import settings
from .routers import (
    admin,
    dashboard,
    dataset,
    training,
    attacks,
    evaluation,
    visualization,
    reports,
    project,
    admin_users,
    invitations,
)
from .routers.auth import router as auth_router

# Windows can report .css as application/x-css. Browsers expect text/css.
mimetypes.add_type("text/css", ".css", strict=True)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="FastAPI backend for the AMCShield research platform.",
)

init_db()

app.add_middleware(
    SessionMiddleware,
    secret_key=settings.session_secret,
    session_cookie="amcshield_session",
    max_age=60 * 60 * 24 * 7,
    same_site="lax",
    https_only=False,  # Local HTTP development. Set True behind HTTPS in production.
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.allowed_origins.strip() != "*",
    allow_methods=["*"],
    allow_headers=["*"],
)

for router in (
    auth_router,

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


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "service": settings.app_name, "version": settings.app_version}


@app.get("/api/health", tags=["Health"])
def api_health():
    return {"status": "ok", "service": settings.app_name, "version": settings.app_version}


# frontend/ is the parent directory of AMCShield_All_In_One.
# StaticFiles serves index.html, CSS, JS and assets without returning
# index.html as a fallback for missing asset paths.
FRONTEND_DIR = settings.frontend_dir.resolve()
if not FRONTEND_DIR.exists():
    raise RuntimeError(f"AMCShield frontend directory not found: {FRONTEND_DIR}")
if not (FRONTEND_DIR / "index.html").is_file():
    raise RuntimeError(f"AMCShield frontend index.html not found: {FRONTEND_DIR / 'index.html'}")

app.mount(
    "/",
    StaticFiles(directory=str(FRONTEND_DIR), html=True),
    name="frontend",
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
