"""
Generation Pipeline Worker

Processes a book into an audiobook:
1. Load chapters and segments from DB
2. Generate audio for each segment using the TTS engine
3. Concatenate segments into chapter audio files
4. Concatenate chapters into full audiobook
5. Export in requested format (MP3/M4B/WAV)

Fails immediately if the TTS engine cannot load.
"""

import logging
import threading
import time
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "storage" / "outputs"
TEMP_DIR = PROJECT_ROOT / "storage" / "temp"

# ---------------------------------------------------------------------------
# Singleton TTS engine -- shared across all worker calls to avoid loading
# the model multiple times (each copy is ~8GB on MPS).
# ---------------------------------------------------------------------------
_tts_engine = None
_tts_engine_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Per-job cancellation tokens.  When a resume is requested the old worker
# thread's event is set so it stops at the next check point.
# ---------------------------------------------------------------------------
_job_cancel_events: dict[str, threading.Event] = {}


def cancel_worker(job_id: str):
    """Signal an in-flight worker for *job_id* to stop at its next checkpoint."""
    ev = _job_cancel_events.get(job_id)
    if ev is not None:
        ev.set()
        logger.info(f"Cancel signal sent to worker for job {job_id}")


def run_generation(job_id: str, is_resume: bool = False):
    """
    Main generation pipeline entry point.
    Called as a background task from the API route.

    Args:
        job_id: UUID of the generation job.
        is_resume: If True, skip already-generated chapters/segments.
    """
    from src.api.database import get_db_context
    from src.api.config import get_config
    from src.models import GenerationJob, Book, Chapter, Segment, VoiceProfile, Character, JobStatus
    from src.audio.concatenator import concatenate_chapters
    from src.audio.splicer import splice_chunks
    from src.tts.chunker import chunk_text, ChunkerConfig
    from src.utils.hardware import profile_resources, log_profile, get_memory_pressure, memory_snapshot, check_memory_safe, get_system_available_gb

    # ---- Cancel any previous worker for this job ----
    old_event = _job_cancel_events.get(job_id)
    if old_event is not None:
        old_event.set()
        logger.warning(f"Signalled previous worker for job {job_id} to stop")

    # Create a fresh cancellation token for this run
    cancel_event = threading.Event()
    _job_cancel_events[job_id] = cancel_event

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    memory_snapshot("worker:startup")

    # ---- Resource profiling ----
    config = get_config()
    rc = config.resources
    profile = profile_resources(
        batch_size_override=rc.batch_size_override,
        memory_headroom_percent=rc.memory_headroom_percent,
        per_segment_cost_gb=rc.per_segment_cost_gb,
        min_batch_size=rc.min_batch_size,
        max_batch_size=rc.max_batch_size,
        flush_interval=rc.flush_interval,
    )
    log_profile(profile)

    base_batch_size = profile.batch_size
    flush_interval = profile.flush_interval
    pressure_threshold = rc.memory_pressure_threshold

    try:
        with get_db_context() as db:
            job = db.query(GenerationJob).filter(GenerationJob.job_id == job_id).first()
            if not job:
                logger.error(f"Job {job_id} not found")
                return

            book = db.query(Book).filter(Book.id == job.book_id).first()
            if not book:
                _fail_job(db, job, "Book not found")
                return

            logger.info(f"{'='*60}")
            logger.info(f"GENERATION {'RESUME' if is_resume else 'START'}: {book.title}")
            logger.info(f"Job: {job_id} | Format: {job.output_format} | Auto: {job.is_auto_mode}")
            if is_resume:
                logger.info(f"Resume #{job.resume_count} | Previously completed: {job.completed_segments} segments")
            logger.info(f"{'='*60}")

            # Update status
            job.status = JobStatus.GENERATING
            if not is_resume:
                job.started_at = datetime.now(timezone.utc)
            job.error_message = None
            db.commit()

            # Load chapters
            chapters = (
                db.query(Chapter)
                .filter(Chapter.book_id == book.id)
                .order_by(Chapter.number)
                .all()
            )
            if not chapters:
                _fail_job(db, job, "No chapters found - book may not be parsed")
                return

            # Count total segments
            total_segments = db.query(Segment).filter(
                Segment.chapter_id.in_([ch.id for ch in chapters])
            ).count()
            job.total_segments = total_segments

            # On resume, initialise completed count from DB state
            if is_resume:
                completed_segments = db.query(Segment).filter(
                    Segment.chapter_id.in_([ch.id for ch in chapters]),
                    Segment.is_generated == True,
                ).count()
                job.completed_segments = completed_segments
                logger.info(f"Resume checkpoint: {completed_segments}/{total_segments} segments already done")
            else:
                completed_segments = 0
            db.commit()

            logger.info(f"Processing {len(chapters)} chapters, {total_segments} segments")

            # Load TTS engine -- singleton, shared across resume calls
            tts_engine = _get_tts_engine()
            if tts_engine is None:
                _fail_job(db, job, "TTS engine failed to load. Check logs for details (model download, MPS issues, etc.)")
                return

            memory_snapshot("worker:post-model-load")

            # Build speaker assignments per character
            speaker_map = _build_speaker_map(db, book.id)
            narrator_speaker = "Ryan"  # Default narrator

            # Process each chapter
            chapter_audio_paths = []
            sample_rate = 24000  # Qwen3 native rate

            for ch_idx, chapter in enumerate(chapters):
                if cancel_event.is_set() or _is_stopped(db, job):
                    logger.info(f"Job {job_id} stopped (cancel_event={cancel_event.is_set()}, status={job.status.value})")
                    return

                # ---- Resume: skip fully completed chapters ----
                if is_resume and chapter.is_generated and chapter.audio_path and Path(chapter.audio_path).exists():
                    chapter_audio_paths.append(chapter.audio_path)
                    logger.info(
                        f"[SKIP] Chapter {chapter.number}/{len(chapters)}: "
                        f"already generated ({chapter.audio_duration_seconds:.1f}s)"
                    )
                    continue

                job.current_chapter = chapter.number
                job.current_step = f"Generating Chapter {chapter.number}/{len(chapters)}: {chapter.title or ''}"
                db.commit()

                logger.info(f"--- Chapter {chapter.number}/{len(chapters)}: {chapter.title or 'Untitled'} ---")

                # Get segments for this chapter
                all_segments = (
                    db.query(Segment)
                    .filter(Segment.chapter_id == chapter.id)
                    .order_by(Segment.sequence_number)
                    .all()
                )

                if not all_segments:
                    logger.warning(f"Chapter {chapter.number} has no segments, skipping")
                    continue

                # On resume, filter to only un-generated segments
                if is_resume:
                    segments = [s for s in all_segments if not s.is_generated]
                    skipped = len(all_segments) - len(segments)
                    if skipped > 0:
                        logger.info(f"  Skipping {skipped} already-generated segments, {len(segments)} remaining")
                    if not segments:
                        # All segments done but chapter not marked -- rebuild chapter audio
                        logger.info(f"  All segments done, rebuilding chapter audio")
                        segments = []  # Fall through to assembly below
                else:
                    segments = all_segments

                # ---- Sentence-level chunked generation ----
                # Build chunking config from app config
                cc = config.chunking
                chunker_cfg = ChunkerConfig(
                    target_seconds=cc.target_chunk_seconds,
                    max_seconds=cc.max_chunk_seconds,
                    words_per_second=cc.words_per_second_estimate,
                )

                segment_audio_paths = []       # Paths to segment WAV files on disk
                paragraph_end_flags = []       # For chapter-level splicing
                BATCH_SEGMENTS = 4             # Process N segments per batch before cleanup

                for batch_start in range(0, len(segments), BATCH_SEGMENTS):
                    # ---- Stop check before each batch ----
                    if cancel_event.is_set() or _is_stopped(db, job):
                        logger.info(
                            f"Job {job_id} stopped mid-chapter "
                            f"(cancel_event={cancel_event.is_set()}, status={job.status.value})"
                        )
                        return

                    batch_segs = segments[batch_start:batch_start + BATCH_SEGMENTS]
                    logger.info(
                        f"--- Batch {batch_start // BATCH_SEGMENTS + 1}: "
                        f"segments {batch_start+1}-{batch_start+len(batch_segs)}/{len(segments)} ---"
                    )

                    # Collect all chunks across this batch of segments,
                    # grouped by speaker for efficient batched TTS calls
                    batch_chunk_groups = []  # list of (seg, chunks, speaker, instruct)
                    for seg in batch_segs:
                        seg_text = seg.user_text_override or seg.text
                        speaker = (
                            speaker_map.get(seg.character_id, narrator_speaker)
                            if seg.character_id else narrator_speaker
                        )
                        instruct = tts_engine._build_instruction({
                            "emotion": seg.emotion, "emphasis": seg.emphasis, "pacing": seg.pacing
                        })
                        chunks = chunk_text(seg_text, config=chunker_cfg)
                        if chunks:
                            batch_chunk_groups.append((seg, chunks, speaker, instruct))

                    if not batch_chunk_groups:
                        continue

                    # Generate each segment's chunks sequentially (batch_size=1)
                    # MPS graph cache leaks per unique tensor shape, so batching
                    # multiple texts (which pads to max length) creates larger
                    # cache entries and leaks faster.  Sequential generation with
                    # cleanup between segment batches is the optimal MPS strategy.
                    total_chunks = sum(len(chunks) for _, chunks, _, _ in batch_chunk_groups)
                    logger.info(
                        f"Generating {total_chunks} chunks from "
                        f"{len(batch_chunk_groups)} segments sequentially"
                    )

                    for seg_i, (seg, chunks, speaker, instruct) in enumerate(batch_chunk_groups):
                        # ---- System-level memory guard ----
                        # Check ACTUAL available system RAM (not just RSS/MPS alloc)
                        # This catches Metal driver allocations invisible to PyTorch
                        safe, avail_gb = check_memory_safe(min_available_gb=16.0)
                        if not safe:
                            logger.warning(
                                f"LOW MEMORY: only {avail_gb:.1f}GB available system RAM. "
                                f"Cycling engine before continuing."
                            )
                            _cycle_tts_engine()
                            tts_engine = _get_tts_engine()
                            safe, avail_gb = check_memory_safe(min_available_gb=8.0)
                            if not safe:
                                _fail_job(db, job,
                                    f"System memory critically low ({avail_gb:.1f}GB available). "
                                    f"Pausing to prevent OOM. Resume when memory is freed."
                                )
                                return

                        try:
                            chunk_results = tts_engine.generate_chunks(
                                chunks=chunks,
                                speaker=speaker,
                                instruct=instruct,
                                temperature=0.7,
                                seed=42,
                            )
                        except Exception as e:
                            _fail_job(db, job, f"TTS failed on ch{chapter.number} seg{seg.sequence_number}: {e}")
                            return

                        chunk_audios = [audio for audio, sr in chunk_results]
                        chunk_para_flags = [c.is_paragraph_end for c in chunks]
                        del chunk_results

                        seg_audio = splice_chunks(
                            chunk_audios=chunk_audios,
                            sample_rate=sample_rate,
                            crossfade_ms=cc.crossfade_ms,
                            sentence_silence_ms=cc.sentence_silence_ms,
                            paragraph_silence_ms=cc.paragraph_silence_ms,
                            target_lufs=-16.0,
                            paragraph_end_flags=chunk_para_flags,
                        )
                        del chunk_audios

                        seg_path = str(
                            TEMP_DIR / f"job_{job_id}_ch{chapter.number:03d}_seg{seg.sequence_number:04d}.wav"
                        )
                        sf.write(seg_path, seg_audio, sample_rate)

                        seg.audio_path = seg_path
                        seg.audio_duration_seconds = len(seg_audio) / sample_rate
                        seg.is_generated = True
                        completed_segments += 1

                        segment_audio_paths.append(seg_path)
                        paragraph_end_flags.append(
                            seg.segment_type.value == "narration"
                        )
                        del seg_audio

                    # ---- Post-batch cleanup: cycle MPS to reclaim graph cache ----
                    _cycle_tts_engine()
                    tts_engine = _get_tts_engine()
                    memory_snapshot(f"worker:post-batch-reset ch{chapter.number}")

                    # Update progress
                    job.completed_segments = completed_segments
                    job.progress = (completed_segments / total_segments) * 100
                    elapsed = (datetime.now(timezone.utc) - (job.resumed_at or job.started_at)).total_seconds()
                    rate = completed_segments / max(elapsed, 1)
                    remaining = (total_segments - completed_segments) / max(rate, 0.001)
                    job.current_step = (
                        f"Chapter {chapter.number}/{len(chapters)} | "
                        f"{completed_segments}/{total_segments} segments | "
                        f"ETA: {int(remaining // 60)}m {int(remaining % 60)}s"
                    )
                    db.commit()

                    logger.info(
                        f"Batch done: {completed_segments}/{total_segments} "
                        f"({job.progress:.1f}%) | "
                        f"rate={rate:.2f} seg/s | ETA={int(remaining)}s"
                    )

                # ---- Assemble chapter from segment audio on disk ----
                if segment_audio_paths:
                    # Read segment audio from disk files (not held in memory)
                    seg_audios = []
                    for sp in segment_audio_paths:
                        audio_data, _ = sf.read(sp)
                        seg_audios.append(audio_data)

                    chapter_audio = splice_chunks(
                        chunk_audios=seg_audios,
                        sample_rate=sample_rate,
                        crossfade_ms=cc.crossfade_ms,
                        sentence_silence_ms=cc.sentence_silence_ms,
                        paragraph_silence_ms=cc.paragraph_silence_ms,
                        target_lufs=-16.0,
                        paragraph_end_flags=paragraph_end_flags,
                    )
                    del seg_audios

                    chapter_path = str(TEMP_DIR / f"job_{job_id}_ch{chapter.number:03d}.wav")
                    sf.write(chapter_path, chapter_audio, sample_rate)

                    chapter.audio_path = chapter_path
                    chapter.audio_duration_seconds = len(chapter_audio) / sample_rate
                    chapter.is_generated = True
                    db.commit()

                    chapter_audio_paths.append(chapter_path)
                    logger.info(
                        f"Chapter {chapter.number} complete: "
                        f"{len(all_segments)} segments, "
                        f"{chapter.audio_duration_seconds:.1f}s audio"
                    )

                    del chapter_audio
                    segment_audio_paths.clear()
                    paragraph_end_flags.clear()

                    memory_snapshot(f"worker:chapter-{chapter.number}-done")

            # Concatenate all chapters into final audiobook
            if not chapter_audio_paths:
                _fail_job(db, job, "No chapter audio generated")
                return

            job.current_step = "Assembling final audiobook..."
            job.progress = 95.0
            db.commit()

            # Build output filename
            safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in book.title)[:50].strip()
            output_filename = f"{safe_title}_{job_id[:8]}.wav"
            output_path = str(OUTPUTS_DIR / output_filename)

            memory_snapshot("worker:pre-final-assembly")
            logger.info(f"Assembling {len(chapter_audio_paths)} chapters into {output_filename}")
            concatenate_chapters(chapter_audio_paths, output_path, sample_rate=sample_rate)
            memory_snapshot("worker:post-final-assembly")

            # Mark complete
            job.status = JobStatus.COMPLETED
            job.progress = 100.0
            job.output_path = output_path
            job.completed_at = datetime.now(timezone.utc)
            job.current_step = "Complete"
            db.commit()

            # Calculate total duration
            output_info = sf.info(output_path)
            total_duration = output_info.duration

            logger.info(f"{'='*60}")
            logger.info(f"GENERATION COMPLETE: {book.title}")
            logger.info(f"Output: {output_path}")
            logger.info(f"Duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
            logger.info(f"Chapters: {len(chapter_audio_paths)} | Segments: {completed_segments}")
            logger.info(f"{'='*60}")

    except Exception as e:
        logger.error(f"Generation failed for job {job_id}: {e}", exc_info=True)
        try:
            with get_db_context() as db:
                job = db.query(GenerationJob).filter(GenerationJob.job_id == job_id).first()
                if job:
                    _fail_job(db, job, str(e))
        except Exception:
            pass
    finally:
        # Clean up cancellation token
        _job_cancel_events.pop(job_id, None)


def _concatenate_part_files(part_paths: list[str], sample_rate: int) -> np.ndarray:
    """Concatenate flushed part WAV files back into a single array."""
    arrays = []
    for p in part_paths:
        audio, sr = sf.read(p)
        if sr != sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        arrays.append(audio)
    return np.concatenate(arrays) if arrays else np.array([], dtype=np.float32)


def _get_tts_engine():
    """Return the singleton TTS engine, loading it on first call.

    The model is ~8GB on MPS. Loading it multiple times (e.g. on resume)
    was the primary cause of the memory doubling issue.
    """
    global _tts_engine
    with _tts_engine_lock:
        if _tts_engine is not None:
            logger.info("Reusing existing TTS engine singleton")
            return _tts_engine

        import psutil
        proc = psutil.Process()
        logger.info(f"Loading TTS engine (first call)... (PID={proc.pid}, RSS={proc.memory_info().rss / (1024**3):.2f}GB)")
        try:
            from src.tts.qwen3_engine import Qwen3TTSEngine
            engine = Qwen3TTSEngine()
            engine._load_model(timeout_seconds=300)
            logger.info(f"TTS engine ready (RSS={proc.memory_info().rss / (1024**3):.2f}GB)")
            _tts_engine = engine
            return _tts_engine
        except Exception as e:
            logger.error(f"Qwen3 TTS engine not available: {e}")
            logger.error("Ensure qwen-tts is installed: pip install -U qwen-tts")
            return None


def _cycle_tts_engine():
    """Unload and reload the TTS engine to reset the MPS memory allocator.

    macOS MPS maps virtual pages for each generate call and never unmaps
    them, causing RSS to grow ~35MB per call.  Destroying and recreating
    the model is the only way to release those pages.  Takes ~4 seconds.
    """
    global _tts_engine
    with _tts_engine_lock:
        if _tts_engine is not None:
            logger.info("Cycling TTS engine to reset MPS allocator...")
            _tts_engine.cleanup()
            _tts_engine = None
            import gc
            gc.collect()
            gc.collect()
            import torch
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
                if hasattr(torch.mps, "synchronize"):
                    torch.mps.synchronize()
            from src.utils.hardware import memory_snapshot
            memory_snapshot("engine:post-unload")


def _load_voice_profiles(db, book_id: int) -> dict:
    """Load voice profiles for all characters in a book."""
    from src.models import Character, VoiceProfile

    profiles = {}
    characters = db.query(Character).filter(Character.book_id == book_id).all()
    for char in characters:
        if char.voice_profile_id:
            profile = db.query(VoiceProfile).filter(VoiceProfile.id == char.voice_profile_id).first()
            if profile:
                profiles[char.id] = {
                    "name": profile.name,
                    "reference_audio_path": profile.reference_audio_path,
                    "temperature": profile.temperature,
                    "seed": profile.seed,
                    "speaking_rate": profile.speaking_rate,
                    "pitch_shift": profile.pitch_shift,
                }

    logger.debug(f"Loaded {len(profiles)} voice profiles for book {book_id}")
    return profiles


def _build_speaker_map(db, book_id: int) -> dict:
    """
    Build a mapping from character_id -> speaker name.
    Assigns preset Qwen3 speakers based on character gender.
    """
    from src.models import Character
    from src.tts.qwen3_engine import Qwen3TTSEngine

    characters = db.query(Character).filter(Character.book_id == book_id).order_by(Character.id).all()
    speaker_map = {}  # character_id -> speaker name
    male_idx = 0
    female_idx = 0

    for char in characters:
        gender = char.inferred_gender
        if gender == "female":
            speaker_map[char.id] = Qwen3TTSEngine.pick_speaker("female", female_idx)
            female_idx += 1
        elif gender == "male":
            speaker_map[char.id] = Qwen3TTSEngine.pick_speaker("male", male_idx)
            male_idx += 1
        else:
            # Alternate for unknown gender
            speaker_map[char.id] = Qwen3TTSEngine.pick_speaker(None, male_idx + female_idx)
            male_idx += 1

    logger.info(f"Speaker assignments: {', '.join(f'{db.query(Character).get(cid).name}={spk}' for cid, spk in list(speaker_map.items())[:10])}")
    return speaker_map


def _is_stopped(db, job) -> bool:
    """Check if the job has been cancelled or paused."""
    from src.models import GenerationJob, JobStatus
    db.refresh(job)
    return job.status in (JobStatus.CANCELLED, JobStatus.PAUSED)


def _fail_job(db, job, error_message: str):
    """Mark a job as failed."""
    from src.models import JobStatus
    job.status = JobStatus.FAILED
    job.error_message = error_message
    job.completed_at = datetime.now(timezone.utc)
    db.commit()
    logger.error(f"Job {job.job_id} [FAILED]: {error_message}")
