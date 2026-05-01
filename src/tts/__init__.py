"""
TTS Module - Qwen3 TTS Engine and Voice Profile Management

Provides context-aware text-to-speech generation with deterministic
voice consistency per character across an entire audiobook.
"""

from .qwen3_engine import Qwen3TTSEngine
from .voice_manager import VoiceProfileManager

__all__ = ["Qwen3TTSEngine", "VoiceProfileManager"]
