"""
Tests for TZDateTime TypeDecorator.

Verifies that datetime values round-tripped through SQLite come back as
timezone-aware (UTC), preventing 'offset-naive vs offset-aware' errors
when subtracting datetimes in the generation worker.
"""

import uuid
import pytest
from datetime import datetime, timezone, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.models.database import Base, GenerationJob, Book, BookFormat, JobStatus


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database for each test."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def sample_book(db_session):
    """Insert a minimal book so we can create generation jobs."""
    book = Book(
        title="Test Book",
        format=BookFormat.TXT,
        file_path="/tmp/test.txt",
        original_filename="test.txt",
        file_size_bytes=100,
    )
    db_session.add(book)
    db_session.commit()
    return book


def test_started_at_round_trip_is_timezone_aware(db_session, sample_book):
    """started_at should remain timezone-aware after commit + re-read."""
    now_utc = datetime.now(timezone.utc)
    job = GenerationJob(
        job_id=str(uuid.uuid4()),
        book_id=sample_book.id,
        status=JobStatus.GENERATING,
        started_at=now_utc,
    )
    db_session.add(job)
    db_session.commit()

    # Expire to force re-read from SQLite (mirrors real session behaviour)
    db_session.expire(job)

    reloaded = job.started_at
    assert reloaded is not None
    assert reloaded.tzinfo is not None, (
        "started_at lost timezone info after round-trip through SQLite"
    )


def test_datetime_subtraction_after_commit(db_session, sample_book):
    """Reproduce the original bug: subtracting aware 'now' from re-read started_at."""
    job = GenerationJob(
        job_id=str(uuid.uuid4()),
        book_id=sample_book.id,
        status=JobStatus.GENERATING,
        started_at=datetime.now(timezone.utc),
    )
    db_session.add(job)
    db_session.commit()
    db_session.expire(job)

    # This line mirrors generation_worker.py:170 and must NOT raise
    elapsed = (datetime.now(timezone.utc) - job.started_at).total_seconds()
    assert elapsed >= 0


def test_created_at_default_is_timezone_aware(db_session, sample_book):
    """created_at (populated by column default) should also be timezone-aware."""
    job = GenerationJob(
        job_id=str(uuid.uuid4()),
        book_id=sample_book.id,
    )
    db_session.add(job)
    db_session.commit()
    db_session.expire(job)

    assert job.created_at is not None
    assert job.created_at.tzinfo is not None, (
        "created_at lost timezone info after round-trip through SQLite"
    )


def test_naive_datetime_is_normalised_on_write(db_session, sample_book):
    """A naive datetime written to a TZDateTime column should get UTC on read."""
    naive = datetime(2025, 6, 15, 12, 0, 0)  # no tzinfo
    job = GenerationJob(
        job_id=str(uuid.uuid4()),
        book_id=sample_book.id,
        started_at=naive,
    )
    db_session.add(job)
    db_session.commit()
    db_session.expire(job)

    reloaded = job.started_at
    assert reloaded.tzinfo is not None
    assert reloaded.tzinfo == timezone.utc


def test_book_created_at_round_trip(db_session):
    """Book.created_at should also survive round-trip as aware."""
    book = Book(
        title="TZ Test",
        format=BookFormat.EPUB,
        file_path="/tmp/tz.epub",
        original_filename="tz.epub",
        file_size_bytes=50,
    )
    db_session.add(book)
    db_session.commit()
    db_session.expire(book)

    assert book.created_at is not None
    assert book.created_at.tzinfo is not None
