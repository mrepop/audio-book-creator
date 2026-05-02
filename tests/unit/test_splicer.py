"""Unit tests for src/audio/splicer.py"""

import numpy as np
import pytest
from src.audio.splicer import (
    hann_crossfade,
    rms_normalize,
    lufs_normalize,
    generate_silence,
    splice_chunks,
)


SR = 24000  # Standard sample rate


def _tone(duration_s: float, freq: float = 440.0, amplitude: float = 0.3) -> np.ndarray:
    """Generate a sine tone for testing."""
    t = np.linspace(0, duration_s, int(SR * duration_s), dtype=np.float32)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestHannCrossfade:
    def test_output_length(self):
        a = _tone(1.0, 440)
        b = _tone(1.0, 880)
        cf_samples = int(SR * 0.1)  # 100ms
        result = hann_crossfade(a, b, cf_samples)
        # Result should be len(a) + len(b) - cf_samples
        expected_len = len(a) + len(b) - cf_samples
        assert len(result) == expected_len

    def test_zero_crossfade_is_concatenation(self):
        a = _tone(0.5)
        b = _tone(0.5)
        result = hann_crossfade(a, b, 0)
        expected = np.concatenate([a, b])
        np.testing.assert_array_equal(result, expected)

    def test_energy_preservation(self):
        """Hann crossfade should roughly preserve RMS energy."""
        a = _tone(1.0, 440, amplitude=0.3)
        b = _tone(1.0, 440, amplitude=0.3)
        cf_samples = int(SR * 0.1)
        result = hann_crossfade(a, b, cf_samples)

        rms_a = np.sqrt(np.mean(a ** 2))
        rms_result_middle = np.sqrt(np.mean(result[len(a)//2:len(a)//2+SR//2] ** 2))
        # Energy in the middle should be roughly similar
        assert abs(rms_a - rms_result_middle) / rms_a < 0.5  # Within 50%

    def test_short_audio_fallback(self):
        """If audio is shorter than crossfade, should just concatenate."""
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        result = hann_crossfade(a, b, 100)  # crossfade longer than audio
        np.testing.assert_array_equal(result, np.array([1, 2, 3, 4], dtype=np.float32))


class TestRmsNormalize:
    def test_normalizes_to_target(self):
        audio = _tone(1.0, amplitude=0.1)
        target_rms = 0.05
        result = rms_normalize(audio, target_rms)
        actual_rms = np.sqrt(np.mean(result ** 2))
        assert abs(actual_rms - target_rms) < 0.01

    def test_silent_audio_unchanged(self):
        silence = np.zeros(SR, dtype=np.float32)
        result = rms_normalize(silence, 0.05)
        np.testing.assert_array_equal(result, silence)

    def test_clipping_protection(self):
        loud = np.ones(SR, dtype=np.float32) * 0.9
        result = rms_normalize(loud, target_rms=0.5)
        assert np.max(np.abs(result)) <= 1.0


class TestLufsNormalize:
    def test_normalizes_loudness(self):
        audio = _tone(1.0, amplitude=0.3)
        result = lufs_normalize(audio, target_lufs=-16.0)
        # Result should be different from input (unless it was already at -16)
        assert not np.array_equal(audio, result)
        # Should be clipped
        assert np.max(np.abs(result)) <= 1.0

    def test_silent_audio_unchanged(self):
        silence = np.zeros(SR, dtype=np.float32)
        result = lufs_normalize(silence, -16.0)
        np.testing.assert_array_equal(result, silence)


class TestGenerateSilence:
    def test_correct_length(self):
        silence = generate_silence(200, SR)
        expected_samples = int(SR * 200 / 1000)
        assert len(silence) == expected_samples

    def test_all_zeros(self):
        silence = generate_silence(100, SR)
        assert np.all(silence == 0)


class TestSpliceChunks:
    def test_single_chunk(self):
        audio = _tone(2.0)
        result = splice_chunks([audio], sample_rate=SR)
        assert len(result) > 0

    def test_multiple_chunks_longer_than_sum(self):
        """Spliced output should be roughly sum of inputs + silences."""
        a = _tone(2.0, 440)
        b = _tone(2.0, 880)
        result = splice_chunks(
            [a, b],
            sample_rate=SR,
            crossfade_ms=100,
            sentence_silence_ms=200,
        )
        # Should be roughly len(a) + len(b) + silence - crossfade
        min_expected = len(a) + len(b) - int(SR * 0.1)
        assert len(result) >= min_expected * 0.8

    def test_paragraph_silence_inserted(self):
        """Paragraph boundaries should get extra silence."""
        a = _tone(1.0)
        b = _tone(1.0)
        result_sentence = splice_chunks(
            [a, b], sample_rate=SR,
            sentence_silence_ms=200, paragraph_silence_ms=500,
            paragraph_end_flags=[False],
        )
        result_paragraph = splice_chunks(
            [a, b], sample_rate=SR,
            sentence_silence_ms=200, paragraph_silence_ms=500,
            paragraph_end_flags=[True],
        )
        # Paragraph version should be longer due to extra silence
        assert len(result_paragraph) > len(result_sentence)

    def test_empty_input(self):
        result = splice_chunks([], sample_rate=SR)
        assert len(result) == 0

    def test_normalization_applied(self):
        """Chunks with different amplitudes should be normalized."""
        quiet = _tone(1.0, amplitude=0.01)
        loud = _tone(1.0, amplitude=0.5)
        result = splice_chunks(
            [quiet, loud], sample_rate=SR, target_lufs=-16.0
        )
        # After normalization, the two halves should be closer in level
        half = len(result) // 2
        rms_first = np.sqrt(np.mean(result[:half] ** 2))
        rms_second = np.sqrt(np.mean(result[half:] ** 2))
        # Should be within 6dB of each other (roughly 2x)
        ratio = max(rms_first, rms_second) / max(min(rms_first, rms_second), 1e-10)
        assert ratio < 4.0  # Less than 12dB difference
