from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_name: str = "AMCShield API"
    app_version: str = "1.0.0"
    environment: str = "development"
    api_prefix: str = "/api"

    frontend_dir: Path = PROJECT_ROOT / "frontend" / "AMCShield_All_In_One"
    src_dir: Path = PROJECT_ROOT / "src"
    results_dir: Path = PROJECT_ROOT / "results"
    checkpoints_dir: Path = PROJECT_ROOT / "checkpoints"
    data_dir: Path = PROJECT_ROOT / "data"

    allowed_origins: str = "http://127.0.0.1:8000,http://localhost:8000"

    # Session cookie security
    session_secret: str = ""
    session_https_only: bool = False
    session_max_age: int = 8 * 60 * 60

    # GitHub OAuth
    github_client_id: str = ""
    github_client_secret: str = ""
    github_redirect_uri: str = "http://127.0.0.1:8000/api/auth/github/callback"
    github_scope: str = "read:user user:email"
    github_state_max_age: int = 600

    # Local administrator
    admin_username: str = "admin"
    admin_email: str = "admin@amcshield.local"
    admin_password: str = ""

    # Account provisioning
    app_base_url: str = "http://127.0.0.1:8000"
    invitation_minutes: int = 60
    password_reset_minutes: int = 30

    # Login protection
    login_max_attempts: int = 5
    login_lockout_minutes: int = 15
    login_attempt_window_minutes: int = 15

    # SMTP
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    smtp_from_name: str = "AMCShield"

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origins(self) -> list[str]:
        return [x.strip() for x in self.allowed_origins.split(",") if x.strip()]


settings = Settings()

if not settings.session_secret:
    raise RuntimeError(
        "SESSION_SECRET must be configured in .env or the environment."
    )

if len(settings.session_secret) < 32:
    raise RuntimeError(
        "SESSION_SECRET must contain at least 32 characters."
    )

if "*" in settings.cors_origins:
    raise RuntimeError(
        "Wildcard CORS origins are not allowed when credentials are enabled."
    )

if settings.environment.strip().lower() == "production":
    if not settings.session_https_only:
        raise RuntimeError(
            "SESSION_HTTPS_ONLY must be true in production."
        )
    if not settings.app_base_url.lower().startswith("https://"):
        raise RuntimeError(
            "APP_BASE_URL must use HTTPS in production."
        )
    if not settings.github_redirect_uri.lower().startswith("https://"):
        raise RuntimeError(
            "GITHUB_REDIRECT_URI must use HTTPS in production."
        )

if settings.github_state_max_age < 60:
    raise RuntimeError("GITHUB_STATE_MAX_AGE must be at least 60 seconds.")
if settings.login_max_attempts < 1:
    raise RuntimeError("LOGIN_MAX_ATTEMPTS must be at least 1.")
if settings.login_lockout_minutes < 1:
    raise RuntimeError("LOGIN_LOCKOUT_MINUTES must be at least 1.")
if settings.session_max_age < 300:
    raise RuntimeError("SESSION_MAX_AGE must be at least 300 seconds.")
