"""
Voice Profile Management Routes
"""

import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from src.api.database import get_db
from src.api.schemas import VoiceProfileCreate, VoiceProfileResponse, VoiceProfileUpdate
from src.models import VoiceProfile

logger = logging.getLogger(__name__)

router = APIRouter()

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
VOICES_DIR = PROJECT_ROOT / "storage" / "voices"


@router.post("/", response_model=VoiceProfileResponse)
async def create_voice_profile(
    profile: VoiceProfileCreate,
    db: Session = Depends(get_db),
):
    """Create a new voice profile."""
    voice = VoiceProfile(**profile.model_dump())
    db.add(voice)
    db.commit()
    db.refresh(voice)
    return voice


@router.get("/", response_model=list[VoiceProfileResponse])
async def list_voice_profiles(db: Session = Depends(get_db)):
    """List all voice profiles."""
    profiles = db.query(VoiceProfile).order_by(VoiceProfile.name).all()
    return profiles


@router.get("/{profile_id}", response_model=VoiceProfileResponse)
async def get_voice_profile(profile_id: int, db: Session = Depends(get_db)):
    """Get a voice profile by ID."""
    profile = db.query(VoiceProfile).filter(VoiceProfile.id == profile_id).first()
    if not profile:
        raise HTTPException(404, "Voice profile not found")
    return profile


@router.patch("/{profile_id}", response_model=VoiceProfileResponse)
async def update_voice_profile(
    profile_id: int,
    update: VoiceProfileUpdate,
    db: Session = Depends(get_db),
):
    """Update voice profile settings."""
    profile = db.query(VoiceProfile).filter(VoiceProfile.id == profile_id).first()
    if not profile:
        raise HTTPException(404, "Voice profile not found")

    for field, value in update.model_dump(exclude_unset=True).items():
        setattr(profile, field, value)

    db.commit()
    db.refresh(profile)
    return profile


@router.post("/{profile_id}/reference-audio")
async def upload_reference_audio(
    profile_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload reference audio for a voice profile."""
    profile = db.query(VoiceProfile).filter(VoiceProfile.id == profile_id).first()
    if not profile:
        raise HTTPException(404, "Voice profile not found")

    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    file_id = str(uuid.uuid4())[:8]
    save_path = VOICES_DIR / f"ref_{profile_id}_{file_id}_{file.filename}"

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    profile.reference_audio_path = str(save_path)
    db.commit()

    return {"status": "uploaded", "path": str(save_path)}


@router.delete("/{profile_id}")
async def delete_voice_profile(profile_id: int, db: Session = Depends(get_db)):
    """Delete a voice profile."""
    profile = db.query(VoiceProfile).filter(VoiceProfile.id == profile_id).first()
    if not profile:
        raise HTTPException(404, "Voice profile not found")

    # Remove reference audio file
    if profile.reference_audio_path:
        ref_path = Path(profile.reference_audio_path)
        if ref_path.exists():
            ref_path.unlink()

    db.delete(profile)
    db.commit()
    return {"status": "deleted", "profile_id": profile_id}
