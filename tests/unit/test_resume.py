"""
Tests for checkpointing and resume support.

Verifies that the PAUSED status, resume tracking columns, and the worker's
resume logic (skip completed chapters, resume partial chapters, correct
progress counters) all work correctly.
"""

import uuid
import pytest
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.models.database import (
    Base, GenerationJob, Book, Chapter, Segment,
    BookFormat, JobStatus, SegmentType,
)


@pytest.fixture
def db_session():
    """In-memory SQLite for each test."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def book_with_chapters(db_session):
    """Book with 3 chapters, 4 segments each."""
    book = Book(
        title="Resume Test Book",
        format=BookFormat.TXT,
        file_path="/tmp/resume.txt",
        original_filename="resume.txt",
        file_size_bytes=100,
        is_parsed=True,
    )
    db_session.add(book)
    db_session.flush()

    for ch_num in range(1, 4):
        chapter = Chapter(
            book_id=book.id,
            number=ch_num,
            title=f"Chapter {ch_num}",
            raw_text=f"Text for chapter {ch_num}.",
            word_count=50,
        )
        db_session.add(chapter)
        db_session.flush()

        for seq in range(1, 5):
            seg = Segment(
                chapter_id=chapter.id,
                text=f"Segment {seq} of chapter {ch_num}.",
                segment_type=SegmentType.NARRATION,
                sequence_number=seq,
            )
            db_session.add(seg)

    db_session.commit()
    return book


# ---- Model tests ----

class TestJobStatusPaused:

    def test_paused_status_exists(self):
        """PAUSED should be a valid JobStatus."""
        assert JobStatus.PAUSED.value == "paused"

    def test_job_can_be_paused(self, db_session, book_with_chapters):
        job = GenerationJob(
            job_id=str(uuid.uuid4()),
            book_id=book_with_chapters.id,
            status=JobStatus.GENERATING,
        )
        db_session.add(job)
        db_session.commit()

        job.status = JobStatus.PAUSED
        db_session.commit()
        db_session.expire(job)

        assert job.status == JobStatus.PAUSED


class TestResumeTracking:

    def test_resume_count_default(self, db_session, book_with_chapters):
        job = GenerationJob(
            job_id=str(uuid.uuid4()),
            book_id=book_with_chapters.id,
        )
        db_session.add(job)
        db_session.commit()
        db_session.expire(job)

        assert job.resume_count == 0
        assert job.resumed_at is None

    def test_resume_count_increments(self, db_session, book_with_chapters):
        job = GenerationJob(
            job_id=str(uuid.uuid4()),
            book_id=book_with_chapters.id,
            status=JobStatus.PAUSED,
        )
        db_session.add(job)
        db_session.commit()

        # Simulate resume
        job.resume_count = (job.resume_count or 0) + 1
        job.resumed_at = datetime.now(timezone.utc)
        job.status = JobStatus.GENERATING
        db_session.commit()
        db_session.expire(job)

        assert job.resume_count == 1
        assert job.resumed_at is not None
        assert job.resumed_at.tzinfo is not None


# ---- Resume logic tests (using DB state, not the full worker) ----

class TestSkipCompletedChapters:

    def test_completed_chapter_detected(self, db_session, book_with_chapters):
        """Chapters with is_generated=True and audio_path should be skippable."""
        chapters = (
            db_session.query(Chapter)
            .filter(Chapter.book_id == book_with_chapters.id)
            .order_by(Chapter.number)
            .all()
        )

        # Mark chapter 1 as complete
        ch1 = chapters[0]
        ch1.is_generated = True
        ch1.audio_path = "/tmp/ch001.wav"
        ch1.audio_duration_seconds = 10.5

        # Mark all its segments as generated
        for seg in ch1.segments:
            seg.is_generated = True
        db_session.commit()

        # Verify skip logic
        skippable = [
            ch for ch in chapters
            if ch.is_generated and ch.audio_path
        ]
        assert len(skippable) == 1
        assert skippable[0].number == 1

        not_done = [ch for ch in chapters if not ch.is_generated]
        assert len(not_done) == 2

    def test_chapter_missing_audio_not_skipped(self, db_session, book_with_chapters):
        """Chapter marked generated but audio file missing should NOT be skipped."""
        ch = (
            db_session.query(Chapter)
            .filter(Chapter.book_id == book_with_chapters.id)
            .first()
        )
        ch.is_generated = True
        ch.audio_path = None  # No audio file
        db_session.commit()

        # Should not be skippable
        assert not (ch.is_generated and ch.audio_path)


class TestPartialChapterResume:

    def test_ungenerated_segments_filtered(self, db_session, book_with_chapters):
        """On resume, only segments with is_generated=False should be processed."""
        ch = (
            db_session.query(Chapter)
            .filter(Chapter.book_id == book_with_chapters.id, Chapter.number == 2)
            .first()
        )
        segments = (
            db_session.query(Segment)
            .filter(Segment.chapter_id == ch.id)
            .order_by(Segment.sequence_number)
            .all()
        )

        # Mark first 2 segments as done
        segments[0].is_generated = True
        segments[1].is_generated = True
        db_session.commit()

        remaining = [s for s in segments if not s.is_generated]
        assert len(remaining) == 2
        assert remaining[0].sequence_number == 3


class TestProgressCounterOnResume:

    def test_completed_count_from_db(self, db_session, book_with_chapters):
        """On resume, completed_segments should be initialized from actual DB state."""
        chapters = (
            db_session.query(Chapter)
            .filter(Chapter.book_id == book_with_chapters.id)
            .all()
        )
        chapter_ids = [ch.id for ch in chapters]

        # Mark 5 of 12 total segments as generated
        segments = (
            db_session.query(Segment)
            .filter(Segment.chapter_id.in_(chapter_ids))
            .order_by(Segment.id)
            .all()
        )
        assert len(segments) == 12  # 3 chapters x 4 segments

        for seg in segments[:5]:
            seg.is_generated = True
        db_session.commit()

        # This is the query the worker runs on resume
        completed = db_session.query(Segment).filter(
            Segment.chapter_id.in_(chapter_ids),
            Segment.is_generated == True,
        ).count()
        assert completed == 5
