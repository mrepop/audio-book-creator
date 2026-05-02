"""
Audio Splicer -- Chunk-Aware Assembly with Crossfade and Normalization

Joins TTS-generated audio chunks into seamless segment/chapter audio.
Each chunk is independently loudness-normalized before joining, and a
Hann-window crossfade is applied at every boundary to eliminate clicks.

Key differences from the old concatenator.py:
  - Per-chunk LUFS normalization *before* joining (consistent levels)
  - Hann window crossfade (perceptually smoother than linear)
  - Separate silence durations for sentence vs paragraph boundaries
  - Designed to work with the Chunk dataclass from chunker.py
"""

import logging
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)


def hann_crossfade(
    audio1: np.ndarray,
    audio2: np.ndarray,
    crossfade_samples: int,
) -> np.ndarray:
    """Join two audio arrays with a Hann-window crossfade.

    Uses cos^2 / sin^2 crossfade (Hann window) which sums to unity,
    preserving energy across the transition.

    Args:
        audio1: First audio array.
        audio2: Second audio array.
        crossfade_samples: Number of samples in the crossfade region.

    Returns:
        Joined audio with smooth transition.
    """
    if crossfade_samples <= 0 or len(audio1) < crossfade_samples or len(audio2) < crossfade_samples:
        return np.concatenate([audio1, audio2])

    t = np.linspace(0, np.pi / 2, crossfade_samples, dtype=np.float32)
    fade_out = np.cos(t) ** 2   # 1 -> 0
    fade_in = np.sin(t) ** 2    # 0 -> 1

    blended = audio1[-crossfade_samples:] * fade_out + audio2[:crossfade_samples] * fade_in

    return np.concatenate([
        audio1[:-crossfade_samples],
        blended,
        audio2[crossfade_samples:],
    ])


def rms_normalize(audio: np.ndarray, target_rms: float = 0.05) -> np.ndarray:
    """Normalize audio to a target RMS level.

    Args:
        audio: Input audio array.
        target_rms: Target RMS amplitude (linear, not dB).

    Returns:
        RMS-normalized audio, clipped to [-1, 1].
    """
    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms < 1e-8:
        return audio
    gain = target_rms / current_rms
    return np.clip(audio * gain, -1.0, 1.0)


def lufs_normalize(audio: np.ndarray, target_lufs: float = -16.0) -> np.ndarray:
    """Simplified LUFS normalization (RMS-based approximation).

    Args:
        audio: Input audio array.
        target_lufs: Target loudness in LUFS.

    Returns:
        Loudness-normalized audio, clipped to [-1, 1].
    """
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-10:
        return audio
    current_lufs = 20 * np.log10(rms + 1e-10)
    gain_db = target_lufs - current_lufs
    gain = 10 ** (gain_db / 20)
    return np.clip(audio * gain, -1.0, 1.0)


def generate_silence(duration_ms: int, sample_rate: int) -> np.ndarray:
    """Generate silence of specified duration."""
    num_samples = int(sample_rate * duration_ms / 1000)
    return np.zeros(num_samples, dtype=np.float32)


def splice_chunks(
    chunk_audios: List[np.ndarray],
    sample_rate: int = 24000,
    crossfade_ms: int = 100,
    sentence_silence_ms: int = 200,
    paragraph_silence_ms: int = 500,
    target_lufs: float = -16.0,
    paragraph_end_flags: Optional[List[bool]] = None,
) -> np.ndarray:
    """Splice chunk audio arrays into a single continuous segment.

    Each chunk is LUFS-normalized independently before joining.
    Hann crossfade is applied at every boundary.  Extra silence is
    inserted after paragraph-ending chunks.

    Args:
        chunk_audios: List of audio arrays (one per chunk).
        sample_rate: Audio sample rate (default 24000 for Qwen3).
        crossfade_ms: Crossfade duration in milliseconds.
        sentence_silence_ms: Silence between sentence chunks.
        paragraph_silence_ms: Silence after paragraph-ending chunks.
        target_lufs: Target loudness for per-chunk normalization.
        paragraph_end_flags: Per-chunk boolean flags indicating paragraph
                             endings.  If None, all boundaries use
                             sentence_silence_ms.

    Returns:
        Spliced audio array.
    """
    if not chunk_audios:
        return np.array([], dtype=np.float32)

    if len(chunk_audios) == 1:
        return lufs_normalize(chunk_audios[0], target_lufs)

    crossfade_samples = int(sample_rate * crossfade_ms / 1000)

    # Normalize each chunk independently
    normalized = []
    for i, audio in enumerate(chunk_audios):
        if len(audio) == 0:
            continue
        norm = lufs_normalize(audio, target_lufs)
        normalized.append(norm)
        logger.debug(
            f"  Chunk {i+1}/{len(chunk_audios)}: "
            f"{len(audio)/sample_rate:.2f}s, normalized to {target_lufs}dB LUFS"
        )

    if not normalized:
        return np.array([], dtype=np.float32)

    # Splice with crossfade + silence
    result = normalized[0]

    for i in range(1, len(normalized)):
        # Determine silence duration
        is_para_end = False
        if paragraph_end_flags and i - 1 < len(paragraph_end_flags):
            is_para_end = paragraph_end_flags[i - 1]

        silence_ms = paragraph_silence_ms if is_para_end else sentence_silence_ms
        silence = generate_silence(silence_ms, sample_rate)

        # Add silence then crossfade into next chunk
        result = np.concatenate([result, silence])
        result = hann_crossfade(result, normalized[i], crossfade_samples)

    total_duration = len(result) / sample_rate
    logger.info(
        f"Spliced {len(normalized)} chunks: {total_duration:.2f}s "
        f"(crossfade={crossfade_ms}ms, LUFS={target_lufs}dB)"
    )

    return result
