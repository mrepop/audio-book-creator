"""
Audio Concatenator

Concatenates audio segments and chapters with crossfading
for seamless audiobook output. Handles silence insertion
between paragraphs and chapters.
"""

import logging
import numpy as np
import soundfile as sf
from typing import List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def crossfade(audio1: np.ndarray, audio2: np.ndarray, fade_samples: int) -> np.ndarray:
    """
    Apply crossfade between two audio arrays.

    Args:
        audio1: First audio segment
        audio2: Second audio segment
        fade_samples: Number of samples for crossfade region

    Returns:
        Crossfaded audio array
    """
    if fade_samples <= 0 or len(audio1) < fade_samples or len(audio2) < fade_samples:
        return np.concatenate([audio1, audio2])

    # Create fade curves
    fade_out = np.linspace(1.0, 0.0, fade_samples)
    fade_in = np.linspace(0.0, 1.0, fade_samples)

    # Apply crossfade
    result = np.copy(audio1)
    result[-fade_samples:] *= fade_out
    blended = result[-fade_samples:] + audio2[:fade_samples] * fade_in

    return np.concatenate([
        audio1[:-fade_samples],
        blended,
        audio2[fade_samples:],
    ])


def generate_silence(duration_ms: int, sample_rate: int) -> np.ndarray:
    """Generate silence of specified duration."""
    num_samples = int(sample_rate * duration_ms / 1000)
    return np.zeros(num_samples, dtype=np.float32)


def concatenate_segments(
    audio_segments: List[np.ndarray],
    sample_rate: int,
    crossfade_ms: int = 150,
    paragraph_silence_ms: int = 500,
) -> np.ndarray:
    """
    Concatenate audio segments with crossfading.

    Args:
        audio_segments: List of audio arrays
        sample_rate: Audio sample rate
        crossfade_ms: Crossfade duration in milliseconds
        paragraph_silence_ms: Silence between segments in milliseconds

    Returns:
        Concatenated audio array
    """
    if not audio_segments:
        return np.array([], dtype=np.float32)

    if len(audio_segments) == 1:
        return audio_segments[0]

    fade_samples = int(sample_rate * crossfade_ms / 1000)
    silence = generate_silence(paragraph_silence_ms, sample_rate)

    result = audio_segments[0]
    for segment in audio_segments[1:]:
        # Add short silence between segments
        result = np.concatenate([result, silence])
        # Apply crossfade
        result = crossfade(result, segment, fade_samples)

    total_duration = len(result) / sample_rate
    logger.info(f"Concatenated {len(audio_segments)} segments: {total_duration:.2f}s")

    return result


def concatenate_chapters(
    chapter_audio_paths: List[str],
    output_path: str,
    sample_rate: int = 24000,
    chapter_silence_ms: int = 2000,
    crossfade_ms: int = 150,
) -> str:
    """
    Concatenate chapter audio files into a full audiobook.

    Args:
        chapter_audio_paths: Ordered list of chapter audio file paths
        output_path: Path to save the full audiobook
        sample_rate: Target sample rate
        chapter_silence_ms: Silence between chapters
        crossfade_ms: Crossfade duration

    Returns:
        Path to the output file
    """
    if not chapter_audio_paths:
        raise ValueError("No chapter audio files provided")

    chapter_silence = generate_silence(chapter_silence_ms, sample_rate)
    fade_samples = int(sample_rate * crossfade_ms / 1000)

    full_audio = None

    for i, path in enumerate(chapter_audio_paths):
        if not Path(path).exists():
            logger.warning(f"Chapter audio not found: {path}")
            continue

        audio, sr = sf.read(path)

        # Resample if necessary
        if sr != sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

        if full_audio is None:
            full_audio = audio
        else:
            full_audio = np.concatenate([full_audio, chapter_silence])
            full_audio = crossfade(full_audio, audio, fade_samples)

        logger.info(f"Added chapter {i + 1}: {len(audio)/sample_rate:.1f}s")

    if full_audio is not None:
        sf.write(output_path, full_audio, sample_rate)
        total_duration = len(full_audio) / sample_rate
        logger.info(f"Full audiobook: {total_duration:.1f}s saved to {output_path}")

    return output_path
