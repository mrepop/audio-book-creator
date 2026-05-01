"""
Audio Processing Module

Handles prosody analysis, audio concatenation, crossfading,
normalization, and final audiobook export.
"""

from .prosody_analyzer import ProsodyAnalyzer, ProsodyFeatures
from .concatenator import concatenate_segments, concatenate_chapters
from .normalizer import normalize_audio, loudness_normalize

__all__ = [
    "ProsodyAnalyzer",
    "ProsodyFeatures",
    "concatenate_segments",
    "concatenate_chapters",
    "normalize_audio",
    "loudness_normalize",
]
