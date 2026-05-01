"""
Audio Normalizer

Handles loudness normalization, peak normalization,
and audio quality adjustments for consistent audiobook output.
"""

import logging
import numpy as np
import soundfile as sf
from typing import Optional

logger = logging.getLogger(__name__)


def normalize_audio(
    audio: np.ndarray,
    target_peak: float = 0.95,
) -> np.ndarray:
    """
    Peak-normalize audio to target level.

    Args:
        audio: Audio array
        target_peak: Target peak amplitude (0.0 to 1.0)

    Returns:
        Normalized audio array
    """
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (target_peak / peak)
    return audio


def loudness_normalize(
    audio: np.ndarray,
    sample_rate: int,
    target_lufs: float = -16.0,
) -> np.ndarray:
    """
    Loudness-normalize audio to target LUFS.

    Uses a simplified loudness estimation based on RMS energy.

    Args:
        audio: Audio array
        sample_rate: Sample rate
        target_lufs: Target loudness in LUFS

    Returns:
        Loudness-normalized audio array
    """
    # Calculate current RMS (approximation of loudness)
    rms = np.sqrt(np.mean(audio ** 2))

    if rms == 0:
        return audio

    # Convert target LUFS to approximate linear gain
    # LUFS ~ 20 * log10(RMS) - 0.691 (simplified)
    current_lufs = 20 * np.log10(rms + 1e-10)
    gain_db = target_lufs - current_lufs
    gain_linear = 10 ** (gain_db / 20)

    # Apply gain with clipping protection
    normalized = audio * gain_linear
    normalized = np.clip(normalized, -1.0, 1.0)

    logger.debug(f"Loudness normalized: {current_lufs:.1f} -> {target_lufs:.1f} LUFS (gain: {gain_db:.1f}dB)")

    return normalized


def normalize_file(
    input_path: str,
    output_path: Optional[str] = None,
    target_lufs: float = -16.0,
) -> str:
    """
    Normalize an audio file.

    Args:
        input_path: Input audio file path
        output_path: Output path (overwrites input if None)
        target_lufs: Target loudness

    Returns:
        Path to normalized audio file
    """
    if output_path is None:
        output_path = input_path

    audio, sr = sf.read(input_path)
    normalized = loudness_normalize(audio, sr, target_lufs)
    normalized = normalize_audio(normalized, target_peak=0.95)
    sf.write(output_path, normalized, sr)

    return output_path
