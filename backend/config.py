from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_name: str = "AMCShield API"
    app_version: str = "1.0.0"
    api_prefix: str = "/api"

    frontend_dir: Path = PROJECT_ROOT / "frontend" / "AMCShield_All_In_One"
    src_dir: Path = PROJECT_ROOT / "src"
    results_dir: Path = PROJECT_ROOT / "results"
    checkpoints_dir: Path = PROJECT_ROOT / "checkpoints"
    data_dir: Path = PROJECT_ROOT / "data"

    allowed_origins: str = "*"

    # Session cookie signing secret.
    # MUST be supplied through .env / environment.
    session_secret: str = ""

    # GitHub OAuth
    github_client_id: str = ""
    github_client_secret: str = ""
    github_redirect_uri: str = (
        "http://127.0.0.1:8000/api/auth/github/callback"
    )
    github_scope: str = "read:user user:email"

    # Local administrator
    admin_username: str = "admin"
    admin_email: str = "admin@amcshield.local"
    admin_password: str = ""

    # Account provisioning
    app_base_url: str = "http://127.0.0.1:8000"
    invitation_minutes: int = 60
    password_reset_minutes: int = 30

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
        if self.allowed_origins.strip() == "*":
            return ["*"]

        return [
            x.strip()
            for x in self.allowed_origins.split(",")
            if x.strip()
        ]


settings = Settings()

if not settings.session_secret:
    raise RuntimeError(
        "SESSION_SECRET must be configured in .env or the environment."
    )