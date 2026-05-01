"""
API Schemas

Pydantic models for request/response validation.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


# ---- Book Schemas ----

class BookUploadResponse(BaseModel):
    id: int
    title: str
    author: Optional[str] = None
    format: str
    original_filename: str
    file_size_bytes: int
    total_chapters: int
    total_words: int
    is_parsed: bool
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class BookDetailResponse(BookUploadResponse):
    language: Optional[str] = None
    total_characters: int = 0
    is_analyzed: bool = False
    chapters: List["ChapterSummaryResponse"] = []
    characters: List["CharacterResponse"] = []


class BookListResponse(BaseModel):
    books: List[BookUploadResponse]
    total: int


# ---- Chapter Schemas ----

class ChapterSummaryResponse(BaseModel):
    id: int
    number: int
    title: Optional[str] = None
    word_count: int
    is_generated: bool
    audio_duration_seconds: Optional[float] = None
    model_config = ConfigDict(from_attributes=True)


class ChapterDetailResponse(ChapterSummaryResponse):
    raw_text: str
    segments: List["SegmentResponse"] = []


# ---- Character Schemas ----

class CharacterResponse(BaseModel):
    id: int
    name: str
    aliases: Optional[str] = None
    description: Optional[str] = None
    role: Optional[str] = None
    inferred_gender: Optional[str] = None
    inferred_age: Optional[str] = None
    dialogue_count: int = 0
    voice_profile_id: Optional[int] = None
    first_appearance_chapter: Optional[int] = None
    model_config = ConfigDict(from_attributes=True)


class CharacterUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    role: Optional[str] = None
    voice_profile_id: Optional[int] = None


# ---- Voice Profile Schemas ----

class VoiceProfileCreate(BaseModel):
    name: str
    description: Optional[str] = None
    gender: Optional[str] = None
    age_range: Optional[str] = None
    voice_quality: Optional[str] = None
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 1.0
    speaking_rate: float = 1.0
    pitch_shift: float = 0.0
    seed: Optional[int] = None


class VoiceProfileResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    gender: Optional[str] = None
    age_range: Optional[str] = None
    voice_quality: Optional[str] = None
    temperature: float
    top_k: int
    top_p: float
    speaking_rate: float
    pitch_shift: float
    seed: Optional[int] = None
    is_default: bool
    reference_audio_path: Optional[str] = None
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class VoiceProfileUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    gender: Optional[str] = None
    age_range: Optional[str] = None
    voice_quality: Optional[str] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    speaking_rate: Optional[float] = None
    pitch_shift: Optional[float] = None
    seed: Optional[int] = None


# ---- Segment Schemas ----

class SegmentResponse(BaseModel):
    id: int
    sequence_number: int
    text: str
    segment_type: str
    character_id: Optional[int] = None
    emotion: Optional[str] = None
    emphasis: Optional[str] = None
    pacing: Optional[str] = None
    is_generated: bool
    audio_duration_seconds: Optional[float] = None
    user_text_override: Optional[str] = None
    user_emphasis_override: Optional[str] = None
    model_config = ConfigDict(from_attributes=True)


class SegmentUpdateRequest(BaseModel):
    """User overrides for a segment."""
    user_text_override: Optional[str] = None
    user_emphasis_override: Optional[str] = None
    user_voice_override_id: Optional[int] = None
    emotion: Optional[str] = None
    pacing: Optional[str] = None


# ---- Generation Job Schemas ----

class GenerationJobCreate(BaseModel):
    book_id: int
    is_auto_mode: bool = True
    output_format: str = "mp3"
    config: Optional[Dict[str, Any]] = None


class GenerationJobResponse(BaseModel):
    id: int
    job_id: str
    book_id: int
    status: str
    progress: float
    current_step: Optional[str] = None
    current_chapter: Optional[int] = None
    total_segments: int
    completed_segments: int
    is_auto_mode: bool
    output_format: str
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


# ---- Dramatis Personae (Character Sheet) ----

class DramatisPersonaeResponse(BaseModel):
    """Complete character sheet for a book."""
    book_id: int
    book_title: str
    characters: List[CharacterResponse]
    total_characters: int
    narrator_voice_profile: Optional[VoiceProfileResponse] = None


# Resolve forward references
BookDetailResponse.model_rebuild()
ChapterDetailResponse.model_rebuild()
