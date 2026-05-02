"""
Qwen3-TTS Engine for Audiobook Generation

Uses the official qwen-tts package with the CustomVoice model for:
- Preset speakers (Ryan, Aiden, Vivian, Serena, etc.)
- Instruction-based emotion/style control
- MPS (Apple Silicon) and CUDA support
- Deterministic generation via temperature/seed control
"""

import logging
from typing import Optional, Tuple, Dict

import torch
import numpy as np

logger = logging.getLogger(__name__)

# Available preset speakers from Qwen3-TTS-CustomVoice
SPEAKERS = {
    "male": ["Ryan", "Aiden"],
    "female": ["Vivian", "Serena"],
    "default": "Ryan",
}

# Emotion -> instruction text for Qwen3 instruct parameter
EMOTION_INSTRUCTIONS = {
    "neutral": "",
    "happy": "Speak with warmth and a cheerful, happy tone.",
    "sad": "Speak with a somber, melancholic tone. Slow and reflective.",
    "angry": "Speak with intensity and controlled anger.",
    "fearful": "Speak with a trembling, anxious, fearful tone.",
    "surprised": "Speak with genuine surprise and astonishment.",
    "tender": "Speak softly and gently, with tenderness.",
    "excited": "Speak with energy and excitement, slightly faster.",
    "contemplative": "Speak thoughtfully and slowly, as if reflecting deeply.",
}

# Emphasis -> instruction modifiers
EMPHASIS_INSTRUCTIONS = {
    "whispered": "Whisper this line very quietly.",
    "shouted": "Shout this line with force and volume.",
    "soft": "Speak very softly and gently.",
    "strong": "Speak firmly and with authority.",
    "normal": "",
}


class Qwen3TTSEngine:
    """
    TTS engine using Qwen3-TTS-12Hz-1.7B-CustomVoice.

    Supports preset speakers with instruction-based emotion control.
    Model weights auto-download from HuggingFace on first load (~3-5GB).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device: Optional[str] = None,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.model_name = model_name
        self._model = None

        # MPS requires float32 and eager attention
        if "mps" in device:
            self._dtype = torch.float32
            self._attn = "eager"
        elif "cuda" in device:
            self._dtype = torch.bfloat16
            # Try flash_attention_2 on CUDA, fall back to eager
            self._attn = "flash_attention_2"
        else:
            self._dtype = torch.float32
            self._attn = "eager"

        logger.info(f"Qwen3-TTS engine configured: device={device}, dtype={self._dtype}, attn={self._attn}")

    def _load_model(self, timeout_seconds: int = 300):
        """Lazy load the TTS model. Downloads weights on first use.

        Args:
            timeout_seconds: Max seconds to wait for model loading (default 5 min).
                             Set to 0 to disable timeout.
        """
        if self._model is not None:
            return self._model

        import time
        import psutil
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

        proc = psutil.Process()
        rss_before = proc.memory_info().rss / (1024 ** 3)

        logger.info(f"Loading Qwen3-TTS model: {self.model_name}")
        logger.info(f"  device={self.device}, dtype={self._dtype}, attn={self._attn}")
        logger.info(f"  RSS before load: {rss_before:.2f}GB")
        logger.info(f"  Timeout: {timeout_seconds}s (0=disabled)")
        logger.info("  (First load will download ~3-5GB of model weights)")

        t0 = time.perf_counter()

        def _do_load(model_name, device, dtype, attn):
            from qwen_tts import Qwen3TTSModel
            logger.info("  [STEP] Calling Qwen3TTSModel.from_pretrained ...")
            model = Qwen3TTSModel.from_pretrained(
                model_name,
                device_map=device,
                dtype=dtype,
                attn_implementation=attn,
            )
            logger.info("  [STEP] from_pretrained returned")
            return model

        try:
            if timeout_seconds > 0:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(_do_load, self.model_name, self.device, self._dtype, self._attn)
                    self._model = future.result(timeout=timeout_seconds)
            else:
                self._model = _do_load(self.model_name, self.device, self._dtype, self._attn)

            elapsed = time.perf_counter() - t0
            rss_after = proc.memory_info().rss / (1024 ** 3)
            logger.info(
                f"[SUCCESS] Qwen3-TTS model loaded in {elapsed:.1f}s | "
                f"RSS: {rss_before:.2f}GB -> {rss_after:.2f}GB (+{rss_after - rss_before:.2f}GB)"
            )

            # Log available speakers
            try:
                speakers = self._model.get_supported_speakers()
                logger.info(f"Available speakers: {speakers}")
            except Exception:
                pass

        except FuturesTimeout:
            elapsed = time.perf_counter() - t0
            logger.error(
                f"[FAILED] Model loading timed out after {elapsed:.0f}s. "
                f"This usually means the model is downloading for the first time "
                f"or MPS is having memory allocation issues."
            )
            raise RuntimeError(f"Qwen3-TTS model loading timed out after {timeout_seconds}s")

        except Exception as e:
            elapsed = time.perf_counter() - t0
            if "flash_attention" in str(e).lower():
                logger.warning(f"FlashAttention not available ({elapsed:.1f}s elapsed), falling back to eager")
                self._attn = "eager"
                self._model = _do_load(self.model_name, self.device, self._dtype, "eager")
                rss_after = proc.memory_info().rss / (1024 ** 3)
                logger.info(
                    f"[SUCCESS] Qwen3-TTS model loaded (eager) in {time.perf_counter() - t0:.1f}s | "
                    f"RSS: {rss_before:.2f}GB -> {rss_after:.2f}GB"
                )
            else:
                logger.error(f"[FAILED] Could not load Qwen3-TTS after {elapsed:.1f}s: {e}")
                raise

        return self._model

    def generate_segment(
        self,
        text: str,
        speaker: str = "Ryan",
        language: str = "English",
        instruct: str = "",
        context: Optional[Dict] = None,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech for a text segment using a preset speaker.

        Args:
            text: Text to synthesize
            speaker: Preset speaker name (Ryan, Aiden, Vivian, Serena, etc.)
            language: Language ("English", "Chinese", etc.)
            instruct: Emotion/style instruction for the model
            context: Context dict with emotion, emphasis, pacing keys
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            seed: Random seed for deterministic generation

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        model = self._load_model()

        # Build instruction from context if not explicitly provided
        if not instruct and context:
            instruct = self._build_instruction(context)

        # Set seed for deterministic generation
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        logger.debug(
            f"TTS: speaker={speaker} | instruct='{instruct[:50]}' | "
            f"temp={temperature} | text='{text[:60]}...'"
        )

        try:
            wavs, sr = model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct if instruct else None,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_new_tokens=2048,
            )

            if not wavs or len(wavs[0]) == 0:
                raise ValueError("Empty waveform returned")

            audio = wavs[0]
            # Move to CPU numpy immediately
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().float().numpy()
            audio = np.array(audio, dtype=np.float32, copy=False)

            logger.debug(f"Generated {len(audio)/sr:.2f}s audio at {sr}Hz")

            del wavs
            self._flush_gpu_cache()

            return audio, sr

        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            self._flush_gpu_cache()
            raise

    def _build_instruction(self, context: Dict) -> str:
        """Build a natural language instruction from context analysis."""
        parts = []

        emotion = context.get("emotion", "neutral")
        if emotion and emotion in EMOTION_INSTRUCTIONS:
            instr = EMOTION_INSTRUCTIONS[emotion]
            if instr:
                parts.append(instr)

        emphasis = context.get("emphasis", "normal")
        if emphasis and emphasis in EMPHASIS_INSTRUCTIONS:
            instr = EMPHASIS_INSTRUCTIONS[emphasis]
            if instr:
                parts.append(instr)

        pacing = context.get("pacing")
        if pacing == "slow":
            parts.append("Speak slowly.")
        elif pacing == "fast":
            parts.append("Speak quickly.")
        elif pacing == "urgent":
            parts.append("Speak with urgency and speed.")

        return " ".join(parts)

    @staticmethod
    def pick_speaker(gender: Optional[str] = None, index: int = 0) -> str:
        """Pick a speaker based on gender and index for variety."""
        if gender == "female":
            speakers = SPEAKERS["female"]
        elif gender == "male":
            speakers = SPEAKERS["male"]
        else:
            speakers = SPEAKERS["male"] + SPEAKERS["female"]
        return speakers[index % len(speakers)]

    def generate_batch(
        self,
        texts: list[str],
        speakers: list[str],
        instructions: list[str],
        language: str = "English",
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> list[tuple[np.ndarray, int]]:
        """
        Generate speech for multiple segments in a single batch.
        Leverages GPU parallelism for significantly faster throughput.

        Args:
            texts: List of texts to synthesize
            speakers: List of speaker names (one per text)
            instructions: List of emotion/style instructions
            language: Language for all segments
            temperature: Sampling temperature

        Returns:
            List of (audio_array, sample_rate) tuples
        """
        model = self._load_model()
        batch_size = len(texts)

        logger.info(f"Batch TTS: {batch_size} segments, speakers={set(speakers)}")

        try:
            # Clean up None instructions to empty strings
            clean_instructions = [inst if inst else "" for inst in instructions]

            wavs, sr = model.generate_custom_voice(
                text=texts,
                language=[language] * batch_size,
                speaker=speakers,
                instruct=clean_instructions if any(clean_instructions) else None,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_new_tokens=2048,
            )

            # Move results to CPU numpy immediately to free GPU/MPS tensors
            results = []
            for i, wav in enumerate(wavs):
                if wav is not None and len(wav) > 0:
                    # Ensure we have a CPU numpy array, not a GPU tensor
                    if isinstance(wav, torch.Tensor):
                        wav = wav.detach().cpu().float().numpy()
                    results.append((np.array(wav, dtype=np.float32, copy=False), sr))
                    logger.debug(f"  Batch [{i+1}/{batch_size}]: {len(wav)/sr:.2f}s")
                else:
                    # Fallback: silence for failed segments
                    word_count = len(texts[i].split())
                    duration = max(0.5, word_count * 0.15)
                    results.append((np.zeros(int(24000 * duration), dtype=np.float32), 24000))
                    logger.warning(f"  Batch [{i+1}/{batch_size}]: empty waveform, using silence")

            # Release all GPU tensors from the model's output
            del wavs
            self._flush_gpu_cache()

            return results

        except Exception as e:
            logger.error(f"Batch TTS failed: {e}", exc_info=True)
            self._flush_gpu_cache()
            raise

    def _flush_gpu_cache(self):
        """Force-release cached GPU/MPS memory back to the OS."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    def cleanup(self):
        """Free model resources."""
        if self._model is not None:
            del self._model
            self._model = None
            self._flush_gpu_cache()
            logger.info("Qwen3 model resources freed")
