"""
Database Connection Management
==============================

SQLAlchemy setup for PostgreSQL connection.
Provides both sync and async session factories.
"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import MetaData, create_engine, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from src.config import settings

# SQLAlchemy Engine
engine = create_engine(
    settings.database_url_sync,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=settings.api_debug,
)

# Session Factory
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)

# Base class for ORM models
Base = declarative_base()

# Metadata for reflection
metadata = MetaData()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI to get database session.

    Yields:
        Session: SQLAlchemy session

    Usage:
        @app.get("/items")
        def read_items(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Context manager for database session.

    Usage:
        with get_db_context() as db:
            db.execute(...)
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def check_connection() -> bool:
    """
    Check if database connection is healthy.

    Returns:
        bool: True if connection is successful
    """
    try:
        with get_db_context() as db:
            db.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def init_db() -> None:
    """
    Initialize database tables.
    Creates all tables defined in Base.metadata.
    """
    Base.metadata.create_all(bind=engine)


def get_table_counts() -> dict:
    """
    Get row counts for all main tables.

    Returns:
        dict: Table names and their row counts
    """
    tables = ["customers", "services", "billing", "churn_labels", "predictions"]
    counts = {}

    with get_db_context() as db:
        for table in tables:
            try:
                result = db.execute(text(f"SELECT COUNT(*) FROM {table}"))
                counts[table] = result.scalar()
            except Exception:
                counts[table] = 0

    return counts
