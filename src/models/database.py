"""
Database Models for Audio Book Creator

Defines all SQLAlchemy ORM models for book management, character tracking,
voice profiles, text segments, and generation jobs.
"""

import enum
import json
from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Text, Float, Boolean, DateTime,
    ForeignKey, Enum, JSON, create_engine
)
from sqlalchemy.orm import declarative_base, relationship, Session
from sqlalchemy.sql import func

Base = declarative_base()


class JobStatus(enum.Enum):
    """Status of a generation job."""
    PENDING = "pending"
    PARSING = "parsing"
    ANALYZING = "analyzing"
    GENERATING = "generating"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SegmentType(enum.Enum):
    """Type of text segment."""
    NARRATION = "narration"
    DIALOGUE = "dialogue"
    INTERNAL_THOUGHT = "internal_thought"
    CHAPTER_TITLE = "chapter_title"
    SCENE_BREAK = "scene_break"


class BookFormat(enum.Enum):
    """Supported ebook formats."""
    EPUB = "epub"
    PDF = "pdf"
    TXT = "txt"


class Book(Base):
    """An uploaded ebook."""
    __tablename__ = "books"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(500), nullable=False)
    author = Column(String(500), nullable=True)
    format = Column(Enum(BookFormat), nullable=False)
    file_path = Column(String(1000), nullable=False)
    original_filename = Column(String(500), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)

    # Metadata extracted from ebook
    language = Column(String(10), default="en")
    total_chapters = Column(Integer, default=0)
    total_characters = Column(Integer, default=0)
    total_words = Column(Integer, default=0)
    metadata_json = Column(Text, nullable=True)  # Additional ebook metadata

    # Processing state
    is_parsed = Column(Boolean, default=False)
    is_analyzed = Column(Boolean, default=False)  # NLP analysis complete
    parsed_at = Column(DateTime, nullable=True)
    analyzed_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    chapters = relationship("Chapter", back_populates="book", cascade="all, delete-orphan")
    characters = relationship("Character", back_populates="book", cascade="all, delete-orphan")
    generation_jobs = relationship("GenerationJob", back_populates="book", cascade="all, delete-orphan")


class Chapter(Base):
    """A chapter within a book."""
    __tablename__ = "chapters"

    id = Column(Integer, primary_key=True, autoincrement=True)
    book_id = Column(Integer, ForeignKey("books.id"), nullable=False)
    number = Column(Integer, nullable=False)
    title = Column(String(500), nullable=True)
    raw_text = Column(Text, nullable=False)
    word_count = Column(Integer, default=0)

    # Audio generation state
    audio_path = Column(String(1000), nullable=True)
    audio_duration_seconds = Column(Float, nullable=True)
    is_generated = Column(Boolean, default=False)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    book = relationship("Book", back_populates="chapters")
    segments = relationship("Segment", back_populates="chapter", cascade="all, delete-orphan")


class VoiceProfile(Base):
    """A reusable voice configuration for TTS generation."""
    __tablename__ = "voice_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False, unique=True)
    description = Column(Text, nullable=True)

    # Voice characteristics
    reference_audio_path = Column(String(1000), nullable=True)
    gender = Column(String(20), nullable=True)  # male, female, neutral
    age_range = Column(String(20), nullable=True)  # child, young, adult, elderly
    voice_quality = Column(String(50), nullable=True)  # warm, gravelly, smooth, etc.

    # TTS generation parameters (deterministic voice settings)
    temperature = Column(Float, default=0.7)
    top_k = Column(Integer, default=50)
    top_p = Column(Float, default=1.0)
    repetition_penalty = Column(Float, default=1.05)
    speaking_rate = Column(Float, default=1.0)
    pitch_shift = Column(Float, default=0.0)
    seed = Column(Integer, nullable=True)  # For deterministic generation

    # Extended settings as JSON
    settings_json = Column(Text, nullable=True)

    is_default = Column(Boolean, default=False)  # Is this the narrator default?
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    characters = relationship("Character", back_populates="voice_profile")

    @property
    def settings(self):
        if self.settings_json:
            return json.loads(self.settings_json)
        return {}


class Character(Base):
    """A character detected in a book."""
    __tablename__ = "characters"

    id = Column(Integer, primary_key=True, autoincrement=True)
    book_id = Column(Integer, ForeignKey("books.id"), nullable=False)
    voice_profile_id = Column(Integer, ForeignKey("voice_profiles.id"), nullable=True)

    name = Column(String(200), nullable=False)
    aliases = Column(Text, nullable=True)  # JSON list of alternate names/references
    description = Column(Text, nullable=True)  # Auto-generated or user-provided description
    role = Column(String(50), nullable=True)  # protagonist, antagonist, supporting, minor

    # Auto-inferred voice traits
    inferred_gender = Column(String(20), nullable=True)
    inferred_age = Column(String(20), nullable=True)
    inferred_personality = Column(Text, nullable=True)

    # Stats
    dialogue_count = Column(Integer, default=0)
    first_appearance_chapter = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    book = relationship("Book", back_populates="characters")
    voice_profile = relationship("VoiceProfile", back_populates="characters")
    segments = relationship("Segment", back_populates="character")


class Segment(Base):
    """A text segment within a chapter, ready for TTS generation."""
    __tablename__ = "segments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chapter_id = Column(Integer, ForeignKey("chapters.id"), nullable=False)
    character_id = Column(Integer, ForeignKey("characters.id"), nullable=True)

    # Segment content
    text = Column(Text, nullable=False)
    segment_type = Column(Enum(SegmentType), nullable=False, default=SegmentType.NARRATION)
    sequence_number = Column(Integer, nullable=False)  # Order within chapter

    # Context analysis results (from NLP module)
    emotion = Column(String(50), nullable=True)  # happy, sad, angry, fearful, etc.
    emphasis = Column(String(50), nullable=True)  # strong, soft, whispered, shouted
    pacing = Column(String(50), nullable=True)  # slow, normal, fast, urgent
    context_json = Column(Text, nullable=True)  # Full context analysis data

    # Audio generation
    audio_path = Column(String(1000), nullable=True)
    audio_duration_seconds = Column(Float, nullable=True)
    is_generated = Column(Boolean, default=False)

    # User overrides
    user_voice_override_id = Column(Integer, ForeignKey("voice_profiles.id"), nullable=True)
    user_emphasis_override = Column(String(50), nullable=True)
    user_text_override = Column(Text, nullable=True)  # User-edited text for this segment

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    chapter = relationship("Chapter", back_populates="segments")
    character = relationship("Character", back_populates="segments")
    user_voice_override = relationship("VoiceProfile", foreign_keys=[user_voice_override_id])


class GenerationJob(Base):
    """A job to generate audio for a book or portion of a book."""
    __tablename__ = "generation_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(36), unique=True, nullable=False)  # UUID
    book_id = Column(Integer, ForeignKey("books.id"), nullable=False)

    # Job configuration
    status = Column(Enum(JobStatus), default=JobStatus.PENDING)
    progress = Column(Float, default=0.0)
    current_step = Column(String(200), nullable=True)
    current_chapter = Column(Integer, nullable=True)
    total_segments = Column(Integer, default=0)
    completed_segments = Column(Integer, default=0)

    # Mode
    is_auto_mode = Column(Boolean, default=True)  # Hands-free mode
    config_json = Column(Text, nullable=True)  # Full generation config

    # Output
    output_format = Column(String(10), default="mp3")
    output_path = Column(String(1000), nullable=True)

    # Timing
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    book = relationship("Book", back_populates="generation_jobs")
