"""
Qwen3-TTS Engine for Audiobook Generation

Adapted from VCS-Audio's Qwen3 engine with audiobook-specific enhancements:
- Context-aware generation (emotion, emphasis, pacing)
- Deterministic voice consistency via seed and temperature control
- Chunked generation for long passages
- Voice profile integration
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict

import torch
import soundfile as sf
import numpy as np

logger = logging.getLogger(__name__)


# Context-to-parameter mappings
EMOTION_TEMPERATURE_MAP = {
    "neutral": 0.7,
    "happy": 0.8,
    "sad": 0.6,
    "angry": 0.85,
    "fearful": 0.75,
    "surprised": 0.8,
    "tender": 0.6,
    "excited": 0.85,
    "contemplative": 0.5,
}

PACING_SPEED_MAP = {
    "normal": 1.0,
    "slow": 0.85,
    "fast": 1.15,
    "urgent": 1.25,
}


class Qwen3TTSEngine:
    """High-quality TTS engine using Qwen3-TTS with audiobook optimizations."""

    def __init__(
        self,
        device: Optional[str] = None,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    ):
        """
        Initialize Qwen3 TTS engine.

        Args:
            device: Device to use ('cpu', 'cuda', 'mps'). Auto-detected if None.
            model_name: HuggingFace model identifier.
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.model_name = model_name
        self._model = None

        logger.info(f"Qwen3-TTS engine initialized (device: {device}, model: {model_name})")

    def _load_model(self):
        """Lazy load the TTS model."""
        if self._model is not None:
            return self._model

        from qwen_tts import Qwen3TTSModel

        logger.info(f"Loading Qwen3-TTS model: {self.model_name}")
        try:
            self._model = Qwen3TTSModel.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=torch.float32,
                attn_implementation="eager",
            )
            logger.info("Qwen3-TTS model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Qwen3-TTS model: {e}")
            raise

        return self._model

    def generate_segment(
        self,
        text: str,
        voice_reference_path: Optional[str] = None,
        context: Optional[Dict] = None,
        output_path: Optional[str] = None,
        # Deterministic voice parameters
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech for a text segment with context-aware parameters.

        Args:
            text: Text to synthesize
            voice_reference_path: Path to voice reference audio
            context: Context dict with emotion, emphasis, pacing keys
            output_path: Optional path to save output WAV
            temperature: Base sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            repetition_penalty: Token repetition penalty
            seed: Random seed for deterministic generation

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        model = self._load_model()

        # Apply context-based parameter adjustments
        if context:
            temperature = self._adjust_temperature(temperature, context)

        # Set seed for deterministic generation
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        if not voice_reference_path:
            raise ValueError("Voice reference audio is required for generation")

        logger.info(f"Generating: '{text[:60]}...' (temp={temperature:.2f}, seed={seed})")

        try:
            wavs, sr = model.generate_voice_clone(
                text=text,
                language="english",
                ref_audio=str(voice_reference_path),
                ref_text=None,
                x_vector_only_mode=True,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                max_new_tokens=2048,
            )

            if not wavs or len(wavs[0]) == 0:
                raise ValueError("Empty waveform returned from Qwen3")

            audio = wavs[0]

            if output_path:
                sf.write(output_path, audio, sr)
                logger.info(f"Saved to: {output_path}")

            logger.info(f"Generated {len(audio)/sr:.2f}s of audio at {sr}Hz")
            return audio, sr

        except Exception as e:
            logger.error(f"Qwen3 generation failed: {e}")
            raise

    def generate_chapter(
        self,
        segments: list,
        voice_profiles: Dict[int, Dict],
        narrator_voice_path: str,
        output_path: str,
        chunk_max_seconds: int = 12,
    ) -> str:
        """
        Generate audio for an entire chapter by processing segments sequentially.

        Args:
            segments: List of Segment ORM objects
            voice_profiles: Dict mapping character_id -> voice settings dict
            narrator_voice_path: Path to narrator voice reference audio
            output_path: Path to save chapter audio
            chunk_max_seconds: Max chunk duration for consistent voice

        Returns:
            Path to generated chapter audio file
        """
        all_audio = []
        sample_rate = None

        for segment in segments:
            # Determine voice settings for this segment
            voice_path = narrator_voice_path
            settings = {"temperature": 0.7, "seed": 42}

            if segment.character_id and segment.character_id in voice_profiles:
                profile = voice_profiles[segment.character_id]
                voice_path = profile.get("reference_audio_path", narrator_voice_path)
                settings = {
                    "temperature": profile.get("temperature", 0.7),
                    "seed": profile.get("seed"),
                    "top_k": profile.get("top_k", 50),
                    "top_p": profile.get("top_p", 1.0),
                }

            # Build context from segment analysis
            context = {
                "emotion": segment.emotion,
                "emphasis": segment.emphasis,
                "pacing": segment.pacing,
            }

            # Use user override text if provided
            text = segment.user_text_override or segment.text

            audio, sr = self.generate_segment(
                text=text,
                voice_reference_path=voice_path,
                context=context,
                **settings,
            )

            all_audio.append(audio)
            if sample_rate is None:
                sample_rate = sr

        # Concatenate all segments
        if all_audio:
            from src.audio.concatenator import concatenate_segments
            full_audio = concatenate_segments(all_audio, sample_rate)
            sf.write(output_path, full_audio, sample_rate)

        return output_path

    def _adjust_temperature(self, base_temp: float, context: Dict) -> float:
        """Adjust temperature based on emotional context."""
        emotion = context.get("emotion", "neutral")
        emotion_temp = EMOTION_TEMPERATURE_MAP.get(emotion, base_temp)

        # Blend base with emotion-adjusted (70% base, 30% emotion)
        adjusted = (base_temp * 0.7) + (emotion_temp * 0.3)
        return round(max(0.3, min(1.0, adjusted)), 2)

    def cleanup(self):
        """Free model resources."""
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Qwen3 model resources freed")
