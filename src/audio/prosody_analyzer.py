"""
Prosody Analyzer

Adapted from VCS-Audio's prosody_analyzer.py.
Extracts prosodic features from audio for quality validation
and TTS parameter tuning.
"""

import logging
import numpy as np
import librosa
from typing import Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProsodyFeatures:
    """Container for prosodic features extracted from audio."""
    speaking_rate: float  # Approximate words per second
    pitch_mean: float  # Mean pitch in Hz
    pitch_std: float  # Pitch variance
    pitch_contour: np.ndarray  # Pitch over time
    energy_mean: float  # Mean energy/loudness
    energy_std: float  # Energy variance
    tempo: float  # Beats per minute
    duration: float  # Audio duration in seconds


class ProsodyAnalyzer:
    """Analyzes prosodic features from audio segments."""

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def extract_features(
        self,
        audio_path: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> ProsodyFeatures:
        """Extract prosodic features from audio file or segment."""
        y, sr = librosa.load(audio_path, sr=self.sample_rate)

        if start_time is not None or end_time is not None:
            start_sample = int(start_time * sr) if start_time else 0
            end_sample = int(end_time * sr) if end_time else len(y)
            y = y[start_sample:end_sample]

        duration = len(y) / sr

        pitch_mean, pitch_std, pitch_contour = self._extract_pitch(y, sr)
        energy_mean, energy_std = self._extract_energy(y)
        tempo = self._estimate_tempo(y, sr)
        speaking_rate = self._estimate_speaking_rate(y, sr)

        return ProsodyFeatures(
            speaking_rate=speaking_rate,
            pitch_mean=pitch_mean,
            pitch_std=pitch_std,
            pitch_contour=pitch_contour,
            energy_mean=energy_mean,
            energy_std=energy_std,
            tempo=tempo,
            duration=duration,
        )

    def _extract_pitch(self, y: np.ndarray, sr: int) -> Tuple[float, float, np.ndarray]:
        """Extract pitch features using pyin."""
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )

        f0_voiced = f0[~np.isnan(f0)]

        if len(f0_voiced) == 0:
            logger.warning("No voiced segments detected")
            return 150.0, 0.0, np.array([150.0])

        pitch_mean = float(np.mean(f0_voiced))
        pitch_std = float(np.std(f0_voiced))

        pitch_contour = np.copy(f0)
        nan_mask = np.isnan(pitch_contour)
        if np.any(nan_mask):
            pitch_contour[nan_mask] = pitch_mean

        return pitch_mean, pitch_std, pitch_contour

    def _extract_energy(self, y: np.ndarray) -> Tuple[float, float]:
        """Extract RMS energy features."""
        rms = librosa.feature.rms(y=y)[0]
        return float(np.mean(rms)), float(np.std(rms))

    def _estimate_tempo(self, y: np.ndarray, sr: int) -> float:
        """Estimate tempo in BPM."""
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            return float(tempo)
        except Exception as e:
            logger.warning(f"Tempo estimation failed: {e}")
            return 120.0

    def _estimate_speaking_rate(self, y: np.ndarray, sr: int) -> float:
        """Estimate speaking rate from zero-crossing rate."""
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        mean_zcr = np.mean(zcr)
        speaking_rate = 2.0 + (mean_zcr * 20.0)
        speaking_rate = np.clip(speaking_rate, 1.5, 5.0)
        return float(speaking_rate)
