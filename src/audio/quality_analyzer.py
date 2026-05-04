"""
Audio Quality Analyzer for TTS Output

Detects common Qwen3-TTS artifacts:
  1. Runaway generation -- audio far longer than expected
  2. Repetition/stuttering -- repeating spectral patterns
  3. Long silence gaps -- unexpected dead air
  4. Clipping -- samples hitting +/- 1.0
  5. Text mismatch -- Whisper transcription vs input text (gibberish detection)

Usage:
    from src.audio.quality_analyzer import analyze_segment
    report = analyze_segment(audio_path, expected_text, expected_duration_s)
    if report["issues"]:
        print("Quality problems:", report["issues"])
"""

import logging
import numpy as np
import soundfile as sf
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-loaded Whisper model (shared across calls)
_whisper_model = None


@dataclass
class QualityReport:
    """Result of audio quality analysis."""
    path: str
    duration_s: float
    expected_duration_s: float
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Metrics
    duration_ratio: float = 0.0        # actual / expected
    silence_ratio: float = 0.0         # fraction of audio that is silence
    max_silence_s: float = 0.0         # longest silent gap
    clipping_ratio: float = 0.0        # fraction of samples at +/- 1.0
    repetition_score: float = 0.0      # 0-1, higher = more repetitive
    text_similarity: float = 1.0       # 0-1, Whisper transcript vs input
    transcription: str = ""            # What Whisper heard

    @property
    def passed(self) -> bool:
        return len(self.issues) == 0

    @property
    def score(self) -> float:
        """Overall quality score 0-100."""
        s = 100.0
        s -= len(self.issues) * 20
        s -= len(self.warnings) * 5
        if self.text_similarity < 1.0:
            s -= (1.0 - self.text_similarity) * 40
        if self.repetition_score > 0.3:
            s -= self.repetition_score * 30
        return max(0.0, min(100.0, s))


def analyze_segment(
    audio_path: str,
    expected_text: str = "",
    expected_duration_s: float = 0.0,
    use_whisper: bool = True,
    whisper_model_size: str = "base",
    # Configurable thresholds (loaded from config.yaml > quality)
    text_similarity_fail: float = 0.6,
    text_similarity_warn: float = 0.75,
    duration_ratio_fail: float = 2.5,
    duration_ratio_warn: float = 1.8,
    max_silence_s_fail: float = 3.0,
    max_silence_s_warn: float = 1.5,
    clipping_ratio_fail: float = 0.01,
    repetition_score_fail: float = 0.7,
) -> QualityReport:
    """Analyze a generated audio segment for quality issues.

    Args:
        audio_path: Path to WAV file.
        expected_text: The text that was supposed to be spoken.
        expected_duration_s: Estimated duration from chunker.
        use_whisper: If True, transcribe with Whisper and compare to expected text.
        whisper_model_size: Whisper model size (tiny, base, small, medium).

    Returns:
        QualityReport with detected issues and metrics.
    """
    audio, sr = sf.read(audio_path)
    duration_s = len(audio) / sr

    report = QualityReport(
        path=audio_path,
        duration_s=duration_s,
        expected_duration_s=expected_duration_s,
    )

    # 1. Duration check
    if expected_duration_s > 0:
        report.duration_ratio = duration_s / expected_duration_s
        if report.duration_ratio > duration_ratio_fail:
            report.issues.append(
                f"RUNAWAY: audio is {duration_s:.1f}s but expected ~{expected_duration_s:.1f}s "
                f"({report.duration_ratio:.1f}x longer)"
            )
        elif report.duration_ratio > duration_ratio_warn:
            report.warnings.append(
                f"LONG: audio is {duration_s:.1f}s, expected ~{expected_duration_s:.1f}s "
                f"({report.duration_ratio:.1f}x)"
            )
        elif report.duration_ratio < 0.3:
            report.warnings.append(
                f"SHORT: audio is {duration_s:.1f}s, expected ~{expected_duration_s:.1f}s"
            )

    # 2. Silence detection
    _check_silence(audio, sr, report, max_silence_s_fail, max_silence_s_warn)

    # 3. Clipping detection
    _check_clipping(audio, report, clipping_ratio_fail)

    # 4. Repetition detection
    _check_repetition(audio, sr, report, repetition_score_fail)

    # 5. Whisper transcription comparison
    if use_whisper and expected_text:
        _check_transcription(audio_path, expected_text, whisper_model_size, report, text_similarity_fail, text_similarity_warn)

    # Log results
    status = "[PASS]" if report.passed else f"[FAIL: {len(report.issues)} issues]"
    logger.info(
        f"Quality {status} {Path(audio_path).name}: "
        f"score={report.score:.0f} duration={duration_s:.1f}s "
        f"silence={report.silence_ratio:.0%} clip={report.clipping_ratio:.1%} "
        f"repeat={report.repetition_score:.2f} text_sim={report.text_similarity:.2f}"
    )
    for issue in report.issues:
        logger.warning(f"  [ISSUE] {issue}")
    for warning in report.warnings:
        logger.info(f"  [WARN] {warning}")

    return report


def _check_silence(audio: np.ndarray, sr: int, report: QualityReport, max_silence_s_fail: float = 3.0, max_silence_s_warn: float = 1.5):
    """Detect long silence gaps."""
    # RMS energy in 50ms windows
    window = int(sr * 0.05)
    if len(audio) < window:
        return

    n_windows = len(audio) // window
    rms = np.array([
        np.sqrt(np.mean(audio[i * window:(i + 1) * window] ** 2))
        for i in range(n_windows)
    ])

    # Silence threshold: -40dB below mean RMS
    mean_rms = np.mean(rms[rms > 0]) if np.any(rms > 0) else 0
    if mean_rms == 0:
        report.issues.append("SILENT: entire audio is silence")
        report.silence_ratio = 1.0
        return

    silence_threshold = mean_rms * 0.01  # -40dB
    is_silent = rms < silence_threshold

    report.silence_ratio = np.mean(is_silent)

    # Find longest contiguous silence
    max_gap = 0
    current_gap = 0
    for s in is_silent:
        if s:
            current_gap += 1
            max_gap = max(max_gap, current_gap)
        else:
            current_gap = 0

    report.max_silence_s = max_gap * 0.05  # window size in seconds

    if report.max_silence_s > max_silence_s_fail:
        report.issues.append(
            f"SILENCE_GAP: {report.max_silence_s:.1f}s silence detected"
        )
    elif report.max_silence_s > max_silence_s_warn:
        report.warnings.append(
            f"SILENCE_GAP: {report.max_silence_s:.1f}s silence"
        )

    if report.silence_ratio > 0.4:
        report.issues.append(
            f"MOSTLY_SILENT: {report.silence_ratio:.0%} of audio is silence"
        )


def _check_clipping(audio: np.ndarray, report: QualityReport, clipping_ratio_fail: float = 0.01):
    """Detect audio clipping."""
    clip_threshold = 0.999
    clipped = np.abs(audio) > clip_threshold
    report.clipping_ratio = np.mean(clipped)

    if report.clipping_ratio > clipping_ratio_fail:
        report.issues.append(
            f"CLIPPING: {report.clipping_ratio:.1%} of samples clipped"
        )
    elif report.clipping_ratio > 0.001:
        report.warnings.append(
            f"MINOR_CLIPPING: {report.clipping_ratio:.2%} of samples"
        )


def _check_repetition(audio: np.ndarray, sr: int, report: QualityReport, repetition_score_fail: float = 0.7):
    """Detect repetitive patterns via normalized autocorrelation.

    Stuttering and repetition loops produce strong peaks in the
    autocorrelation at regular intervals.
    """
    # Work on 1-second chunks, check for periodic repetition
    chunk_s = 1.0
    chunk_samples = int(sr * chunk_s)

    if len(audio) < chunk_samples * 4:
        report.repetition_score = 0.0
        return

    # Compute energy per chunk
    n_chunks = len(audio) // chunk_samples
    energies = np.array([
        np.sqrt(np.mean(audio[i * chunk_samples:(i + 1) * chunk_samples] ** 2))
        for i in range(n_chunks)
    ])

    if len(energies) < 4 or np.std(energies) < 1e-8:
        report.repetition_score = 0.0
        return

    # Normalized autocorrelation of energy sequence
    energies_norm = (energies - np.mean(energies)) / (np.std(energies) + 1e-10)
    autocorr = np.correlate(energies_norm, energies_norm, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]  # positive lags only
    autocorr = autocorr / (autocorr[0] + 1e-10)  # normalize

    # Look for strong peaks at lags 2-10 (2-10 second repetition period)
    if len(autocorr) > 10:
        peak_value = np.max(autocorr[2:min(10, len(autocorr))])
        report.repetition_score = max(0.0, float(peak_value))
    else:
        report.repetition_score = 0.0

    if report.repetition_score > repetition_score_fail:
        report.issues.append(
            f"REPETITION: strong repeating pattern detected "
            f"(score={report.repetition_score:.2f})"
        )
    elif report.repetition_score > 0.5:
        report.warnings.append(
            f"POSSIBLE_REPETITION: moderate pattern "
            f"(score={report.repetition_score:.2f})"
        )


def _check_transcription(
    audio_path: str,
    expected_text: str,
    model_size: str,
    report: QualityReport,
    text_similarity_fail: float = 0.6,
    text_similarity_warn: float = 0.75,
):
    """Transcribe with Whisper and compare to expected text."""
    global _whisper_model

    try:
        import whisper

        if _whisper_model is None:
            logger.info(f"Loading Whisper {model_size} model...")
            _whisper_model = whisper.load_model(model_size)

        result = _whisper_model.transcribe(
            audio_path,
            language="en",
            fp16=False,  # MPS doesn't support fp16 for Whisper
        )
        transcription = result["text"].strip()
        report.transcription = transcription

        # Compare using SequenceMatcher (word-level similarity)
        expected_words = expected_text.lower().split()
        actual_words = transcription.lower().split()

        if not expected_words:
            report.text_similarity = 1.0
            return

        matcher = SequenceMatcher(None, expected_words, actual_words)
        report.text_similarity = matcher.ratio()

        if report.text_similarity < text_similarity_fail:
            report.issues.append(
                f"GIBBERISH: transcription doesn't match input "
                f"(similarity={report.text_similarity:.0%}). "
                f"Expected: '{expected_text[:80]}...' "
                f"Got: '{transcription[:80]}...'"
            )
        elif report.text_similarity < text_similarity_warn:
            report.warnings.append(
                f"MISMATCH: partial text match "
                f"(similarity={report.text_similarity:.0%}). "
                f"Got: '{transcription[:80]}...'"
            )

    except Exception as e:
        logger.warning(f"Whisper transcription failed: {e}")
        report.warnings.append(f"WHISPER_FAILED: {e}")


def analyze_directory(
    directory: str,
    expected_texts: Optional[dict[str, str]] = None,
    use_whisper: bool = True,
) -> list[QualityReport]:
    """Analyze all WAV files in a directory.

    Args:
        directory: Path to directory containing WAV files.
        expected_texts: Optional dict mapping filename -> expected text.
        use_whisper: Whether to use Whisper for transcription checks.

    Returns:
        List of QualityReport objects.
    """
    reports = []
    wav_files = sorted(Path(directory).glob("*.wav"))

    logger.info(f"Analyzing {len(wav_files)} WAV files in {directory}")

    for wav_path in wav_files:
        expected = ""
        if expected_texts and wav_path.name in expected_texts:
            expected = expected_texts[wav_path.name]

        report = analyze_segment(
            str(wav_path),
            expected_text=expected,
            use_whisper=use_whisper,
        )
        reports.append(report)

    # Summary
    passed = sum(1 for r in reports if r.passed)
    failed = len(reports) - passed
    avg_score = sum(r.score for r in reports) / max(len(reports), 1)

    logger.info(f"Quality analysis complete: {passed} passed, {failed} failed, avg score={avg_score:.0f}")

    return reports
