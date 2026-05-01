"""
Voice Profile Manager

Manages voice profiles for consistent character voices throughout
an audiobook. Handles reference audio, deterministic seed management,
and voice parameter persistence.
"""

import logging
import json
from typing import Dict, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class VoiceProfileManager:
    """
    Manages voice profiles for consistent character voices.

    Ensures deterministic voice generation by maintaining:
    - Fixed reference audio per character
    - Consistent seed values
    - Stable temperature and generation parameters
    """

    def __init__(self, db_session=None):
        self._db = db_session
        self._profile_cache: Dict[int, Dict] = {}

    def get_profile_settings(self, profile_id: int) -> Dict:
        """
        Get voice generation settings for a profile.

        Returns dict with: reference_audio_path, temperature, seed,
        top_k, top_p, speaking_rate, pitch_shift
        """
        if profile_id in self._profile_cache:
            return self._profile_cache[profile_id]

        if self._db is None:
            return self._default_settings()

        from src.models import VoiceProfile
        profile = self._db.query(VoiceProfile).filter(VoiceProfile.id == profile_id).first()

        if not profile:
            logger.warning(f"Voice profile {profile_id} not found, using defaults")
            return self._default_settings()

        settings = {
            "reference_audio_path": profile.reference_audio_path,
            "temperature": profile.temperature,
            "top_k": profile.top_k,
            "top_p": profile.top_p,
            "repetition_penalty": profile.repetition_penalty,
            "speaking_rate": profile.speaking_rate,
            "pitch_shift": profile.pitch_shift,
            "seed": profile.seed,
            "gender": profile.gender,
            "age_range": profile.age_range,
            "voice_quality": profile.voice_quality,
        }

        self._profile_cache[profile_id] = settings
        return settings

    def get_all_book_profiles(self, book_id: int) -> Dict[int, Dict]:
        """
        Load all voice profiles for characters in a book.

        Returns dict mapping character_id -> voice settings
        """
        if self._db is None:
            return {}

        from src.models import Character
        characters = (
            self._db.query(Character)
            .filter(Character.book_id == book_id)
            .all()
        )

        profiles = {}
        for char in characters:
            if char.voice_profile_id:
                profiles[char.id] = self.get_profile_settings(char.voice_profile_id)

        logger.info(f"Loaded {len(profiles)} voice profiles for book {book_id}")
        return profiles

    def auto_assign_voices(self, book_id: int) -> Dict[int, int]:
        """
        Auto-assign voice profiles to characters based on inferred traits.

        Creates new voice profiles for characters that don't have one,
        using inferred gender and role to set initial parameters.

        Returns dict mapping character_id -> voice_profile_id
        """
        if self._db is None:
            return {}

        from src.models import Character, VoiceProfile

        characters = (
            self._db.query(Character)
            .filter(Character.book_id == book_id)
            .all()
        )

        assignments = {}
        seed_base = 42  # Deterministic seed base

        for i, char in enumerate(characters):
            if char.voice_profile_id:
                assignments[char.id] = char.voice_profile_id
                continue

            # Create a voice profile based on inferred traits
            profile = VoiceProfile(
                name=f"{char.name} (Auto)",
                description=f"Auto-generated voice for {char.name}",
                gender=char.inferred_gender,
                age_range=char.inferred_age,
                temperature=self._trait_temperature(char),
                seed=seed_base + i,  # Unique but deterministic seed per character
                speaking_rate=self._trait_speaking_rate(char),
                pitch_shift=self._trait_pitch_shift(char),
            )
            self._db.add(profile)
            self._db.flush()

            char.voice_profile_id = profile.id
            assignments[char.id] = profile.id

        self._db.commit()
        logger.info(f"Auto-assigned {len(assignments)} voice profiles for book {book_id}")
        return assignments

    def _default_settings(self) -> Dict:
        """Default narrator voice settings."""
        return {
            "reference_audio_path": None,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 1.0,
            "repetition_penalty": 1.05,
            "speaking_rate": 1.0,
            "pitch_shift": 0.0,
            "seed": 42,
        }

    def _trait_temperature(self, character) -> float:
        """Set temperature based on character traits."""
        role = character.role or "minor"
        if role == "protagonist":
            return 0.6  # More consistent for main character
        elif role == "supporting":
            return 0.65
        return 0.7

    def _trait_speaking_rate(self, character) -> float:
        """Set speaking rate based on character traits."""
        age = character.inferred_age
        if age == "elderly":
            return 0.9
        elif age == "child":
            return 1.1
        return 1.0

    def _trait_pitch_shift(self, character) -> float:
        """Set pitch shift based on inferred traits."""
        gender = character.inferred_gender
        if gender == "female":
            return 2.0  # Slightly higher pitch
        elif gender == "male":
            return -2.0  # Slightly lower pitch
        return 0.0
