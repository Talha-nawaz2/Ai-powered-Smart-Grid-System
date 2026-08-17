"""
Database engine + session management.

Concept: This is the ONLY file that knows how to connect to the database.
Every other module asks THIS file for a session rather than opening its own
connection. That's what makes it easy to swap SQLite -> PostgreSQL later:
you change DATABASE_URL in config.py and nothing else in the codebase moves.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from backend.utils.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},  # needed for SQLite + FastAPI
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency: yields a DB session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
