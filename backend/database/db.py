from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import Base


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DB_PATH = (
    PROJECT_ROOT
    / "data"
    / "amcshield_auth.db"
)

DB_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


DATABASE_URL = f"sqlite:///{DB_PATH.as_posix()}"


engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False
    },
    future=True,
)


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


def init_db():
    Base.metadata.create_all(
        bind=engine
    )


def get_db():
    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()