"""
Audiobook Generation Routes

Create and manage generation jobs for converting books to audio.
"""

import json
import uuid
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from src.api.database import get_db
from src.api.schemas import (
    GenerationJobCreate, GenerationJobResponse,
    SegmentResponse, SegmentUpdateRequest,
)
from src.models import GenerationJob, Book, Segment, JobStatus

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/jobs", response_model=GenerationJobResponse)
async def create_generation_job(
    request: GenerationJobCreate,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db),
):
    """Create a new audiobook generation job."""
    book = db.query(Book).filter(Book.id == request.book_id).first()
    if not book:
        raise HTTPException(404, "Book not found")
    if not book.is_parsed:
        raise HTTPException(400, "Book must be parsed before generation")

    job = GenerationJob(
        job_id=str(uuid.uuid4()),
        book_id=request.book_id,
        is_auto_mode=request.is_auto_mode,
        output_format=request.output_format,
        config_json=json.dumps(request.config) if request.config else None,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Start generation in background
    background_tasks.add_task(_run_generation, job.job_id)

    logger.info(f"Generation job created: {job.job_id} for book {book.title}")
    return job


@router.get("/jobs", response_model=list[GenerationJobResponse])
async def list_generation_jobs(
    book_id: int = None,
    db: Session = Depends(get_db),
):
    """List generation jobs, optionally filtered by book."""
    query = db.query(GenerationJob)
    if book_id:
        query = query.filter(GenerationJob.book_id == book_id)
    jobs = query.order_by(GenerationJob.created_at.desc()).all()
    return jobs


@router.get("/jobs/{job_id}", response_model=GenerationJobResponse)
async def get_generation_job(job_id: str, db: Session = Depends(get_db)):
    """Get generation job status."""
    job = db.query(GenerationJob).filter(GenerationJob.job_id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@router.post("/jobs/{job_id}/cancel")
async def cancel_generation_job(job_id: str, db: Session = Depends(get_db)):
    """Cancel a running generation job."""
    job = db.query(GenerationJob).filter(GenerationJob.job_id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
        raise HTTPException(400, f"Job already in terminal state: {job.status.value}")

    job.status = JobStatus.CANCELLED
    db.commit()
    return {"status": "cancelled", "job_id": job_id}


@router.get("/jobs/{job_id}/download")
async def download_audiobook(job_id: str, db: Session = Depends(get_db)):
    """Download the generated audiobook file."""
    job = db.query(GenerationJob).filter(GenerationJob.job_id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(400, "Job is not completed")
    if not job.output_path or not Path(job.output_path).exists():
        raise HTTPException(404, "Output file not found")

    return FileResponse(
        job.output_path,
        media_type="audio/mpeg",
        filename=Path(job.output_path).name,
    )


# ---- Segment-level editing ----

@router.get("/chapters/{chapter_id}/segments", response_model=list[SegmentResponse])
async def list_segments(chapter_id: int, db: Session = Depends(get_db)):
    """List all segments in a chapter for sentence-level editing."""
    segments = (
        db.query(Segment)
        .filter(Segment.chapter_id == chapter_id)
        .order_by(Segment.sequence_number)
        .all()
    )
    return segments


@router.patch("/segments/{segment_id}", response_model=SegmentResponse)
async def update_segment(
    segment_id: int,
    update: SegmentUpdateRequest,
    db: Session = Depends(get_db),
):
    """Update a segment (user overrides for text, emphasis, voice, etc.)."""
    segment = db.query(Segment).filter(Segment.id == segment_id).first()
    if not segment:
        raise HTTPException(404, "Segment not found")

    for field, value in update.model_dump(exclude_unset=True).items():
        setattr(segment, field, value)

    db.commit()
    db.refresh(segment)
    return segment


@router.post("/segments/{segment_id}/regenerate")
async def regenerate_segment(
    segment_id: int,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db),
):
    """Regenerate audio for a single segment."""
    segment = db.query(Segment).filter(Segment.id == segment_id).first()
    if not segment:
        raise HTTPException(404, "Segment not found")

    segment.is_generated = False
    db.commit()

    background_tasks.add_task(_regenerate_segment, segment_id)
    return {"status": "regenerating", "segment_id": segment_id}


async def _run_generation(job_id: str):
    """Background task: run full audiobook generation pipeline."""
    from src.api.database import get_db_context

    try:
        with get_db_context() as db:
            job = db.query(GenerationJob).filter(GenerationJob.job_id == job_id).first()
            if not job:
                return

            job.status = JobStatus.GENERATING
            db.commit()

            # TODO: Implement full generation pipeline
            # 1. For each chapter, get segments
            # 2. For each segment, generate audio with appropriate voice
            # 3. Concatenate segments into chapter audio
            # 4. Concatenate chapters into full audiobook
            # 5. Export in requested format

            logger.info(f"Generation job {job_id} started (pipeline not yet implemented)")
    except Exception as e:
        logger.error(f"Generation job {job_id} failed: {e}")


async def _regenerate_segment(segment_id: int):
    """Background task: regenerate a single segment's audio."""
    from src.api.database import get_db_context

    try:
        with get_db_context() as db:
            segment = db.query(Segment).filter(Segment.id == segment_id).first()
            if not segment:
                return

            # TODO: Implement single segment regeneration
            logger.info(f"Segment {segment_id} regeneration started (not yet implemented)")
    except Exception as e:
        logger.error(f"Segment {segment_id} regeneration failed: {e}")
