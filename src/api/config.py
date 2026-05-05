"""
Configuration Management

Loads configuration from config.yaml with environment variable overrides
and platform-specific settings. Follows VCS-Audio config pattern.
"""

import os
import sys
import platform
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import yaml

logger = logging.getLogger(__name__)

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    timeout_keep_alive: int = 300


@dataclass
class TTSConfig:
    engine: str = "qwen3"
    model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    default_temperature: float = 0.7
    default_top_k: int = 50
    default_top_p: float = 1.0
    repetition_penalty: float = 1.05
    max_new_tokens: int = 2048
    chunk_max_seconds: int = 12
    voice_sample_seconds: int = 20


@dataclass
class AudioConfig:
    sample_rate: int = 24000
    output_sample_rate: int = 44100
    crossfade_ms: int = 150
    paragraph_silence_ms: int = 500
    chapter_silence_ms: int = 2000
    normalization_target_db: int = -16
    output_formats: list = field(default_factory=lambda: ["mp3", "m4b", "wav"])


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "detailed"
    file_output: bool = True
    log_dir: str = "logs"


@dataclass
class ChunkingConfig:
    target_chunk_seconds: float = 12.0
    max_chunk_seconds: float = 15.0
    sentence_silence_ms: int = 200
    paragraph_silence_ms: int = 500
    crossfade_ms: int = 100
    words_per_second_estimate: float = 2.5


@dataclass
class QualityConfig:
    enabled: bool = True
    use_whisper: bool = True
    whisper_model: str = "base"
    min_words_for_whisper: int = 5
    whisper_parallel: bool = True
    max_retries: int = 2
    retry_log_level: str = "WARNING"
    failure_log_level: str = "ERROR"
    text_similarity_fail: float = 0.6
    text_similarity_warn: float = 0.75
    duration_ratio_fail: float = 2.5
    duration_ratio_warn: float = 1.8
    max_silence_s_fail: float = 3.0
    max_silence_s_warn: float = 1.5
    clipping_ratio_fail: float = 0.01
    repetition_score_fail: float = 0.7


@dataclass
class ResourceConfig:
    batch_size: str = "auto"  # "auto" or integer string
    memory_headroom_percent: float = 20.0
    per_segment_cost_gb: float = 1.5
    max_batch_size: int = 12
    min_batch_size: int = 1
    flush_interval: int = 50
    memory_pressure_threshold: float = 0.85
    engine_pool_size: str = "auto"  # "auto" or integer string

    @property
    def pool_size_override(self) -> Optional[int]:
        if self.engine_pool_size == "auto":
            return None
        try:
            return int(self.engine_pool_size)
        except (ValueError, TypeError):
            return None

    @property
    def batch_size_override(self) -> Optional[int]:
        """Return int if manually set, None if 'auto'."""
        if self.batch_size == "auto":
            return None
        try:
            return int(self.batch_size)
        except (ValueError, TypeError):
            return None


@dataclass
class NLPConfig:
    spacy_model: str = "en_core_web_sm"
    llm_enabled: bool = True
    llm_model: str = "mlx-community/Qwen2.5-7B-Instruct-4bit"
    llm_context_window: int = 12
    llm_context_overlap: int = 3
    emotion_detection: bool = True
    speaker_attribution: bool = True
    auto_voice_traits: bool = True


@dataclass
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    config_path: Optional[str] = None


def get_platform_info() -> dict:
    """Detect platform, GPU availability, and system capabilities."""
    info = {
        "os": platform.system().lower(),
        "arch": platform.machine(),
        "python_version": platform.python_version(),
        "gpu_type": "cpu",
        "gpu_available": False,
        "torch_version": None,
    }

    if _TORCH_AVAILABLE:
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu_type"] = "cuda"
            info["gpu_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["gpu_type"] = "mps"
            info["gpu_available"] = True
            info["gpu_name"] = "Apple Silicon"

    return info


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load configuration from YAML file with env var overrides."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"

    config = AppConfig()

    # Load YAML if it exists
    if Path(config_path).exists():
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f) or {}

        # Server
        srv = raw.get("server", {})
        config.server = ServerConfig(
            host=srv.get("host", config.server.host),
            port=srv.get("port", config.server.port),
            reload=srv.get("reload", config.server.reload),
            timeout_keep_alive=srv.get("timeout_keep_alive", config.server.timeout_keep_alive),
        )

        # TTS
        tts_raw = raw.get("tts", {})
        qwen_raw = tts_raw.get("qwen3", {})
        config.tts = TTSConfig(
            engine=tts_raw.get("engine", config.tts.engine),
            model_name=qwen_raw.get("model_name", config.tts.model_name),
            default_temperature=qwen_raw.get("default_temperature", config.tts.default_temperature),
            default_top_k=qwen_raw.get("default_top_k", config.tts.default_top_k),
            default_top_p=qwen_raw.get("default_top_p", config.tts.default_top_p),
            repetition_penalty=qwen_raw.get("repetition_penalty", config.tts.repetition_penalty),
            max_new_tokens=qwen_raw.get("max_new_tokens", config.tts.max_new_tokens),
            chunk_max_seconds=qwen_raw.get("chunk_max_seconds", config.tts.chunk_max_seconds),
            voice_sample_seconds=qwen_raw.get("voice_sample_seconds", config.tts.voice_sample_seconds),
        )

        # Audio
        aud = raw.get("audio", {})
        config.audio = AudioConfig(
            sample_rate=aud.get("sample_rate", config.audio.sample_rate),
            output_sample_rate=aud.get("output_sample_rate", config.audio.output_sample_rate),
            crossfade_ms=aud.get("crossfade_ms", config.audio.crossfade_ms),
            paragraph_silence_ms=aud.get("paragraph_silence_ms", config.audio.paragraph_silence_ms),
            chapter_silence_ms=aud.get("chapter_silence_ms", config.audio.chapter_silence_ms),
            normalization_target_db=aud.get("normalization_target_db", config.audio.normalization_target_db),
            output_formats=aud.get("output_formats", config.audio.output_formats),
        )

        # Chunking
        chunk_raw = raw.get("chunking", {})
        config.chunking = ChunkingConfig(
            target_chunk_seconds=float(chunk_raw.get("target_chunk_seconds", config.chunking.target_chunk_seconds)),
            max_chunk_seconds=float(chunk_raw.get("max_chunk_seconds", config.chunking.max_chunk_seconds)),
            sentence_silence_ms=int(chunk_raw.get("sentence_silence_ms", config.chunking.sentence_silence_ms)),
            paragraph_silence_ms=int(chunk_raw.get("paragraph_silence_ms", config.chunking.paragraph_silence_ms)),
            crossfade_ms=int(chunk_raw.get("crossfade_ms", config.chunking.crossfade_ms)),
            words_per_second_estimate=float(chunk_raw.get("words_per_second_estimate", config.chunking.words_per_second_estimate)),
        )

        # Quality
        qa_raw = raw.get("quality", {})
        config.quality = QualityConfig(
            enabled=bool(qa_raw.get("enabled", config.quality.enabled)),
            use_whisper=bool(qa_raw.get("use_whisper", config.quality.use_whisper)),
            whisper_model=str(qa_raw.get("whisper_model", config.quality.whisper_model)),
            min_words_for_whisper=int(qa_raw.get("min_words_for_whisper", config.quality.min_words_for_whisper)),
            whisper_parallel=bool(qa_raw.get("whisper_parallel", config.quality.whisper_parallel)),
            max_retries=int(qa_raw.get("max_retries", config.quality.max_retries)),
            retry_log_level=str(qa_raw.get("retry_log_level", config.quality.retry_log_level)),
            failure_log_level=str(qa_raw.get("failure_log_level", config.quality.failure_log_level)),
            text_similarity_fail=float(qa_raw.get("text_similarity_fail", config.quality.text_similarity_fail)),
            text_similarity_warn=float(qa_raw.get("text_similarity_warn", config.quality.text_similarity_warn)),
            duration_ratio_fail=float(qa_raw.get("duration_ratio_fail", config.quality.duration_ratio_fail)),
            duration_ratio_warn=float(qa_raw.get("duration_ratio_warn", config.quality.duration_ratio_warn)),
            max_silence_s_fail=float(qa_raw.get("max_silence_s_fail", config.quality.max_silence_s_fail)),
            max_silence_s_warn=float(qa_raw.get("max_silence_s_warn", config.quality.max_silence_s_warn)),
            clipping_ratio_fail=float(qa_raw.get("clipping_ratio_fail", config.quality.clipping_ratio_fail)),
            repetition_score_fail=float(qa_raw.get("repetition_score_fail", config.quality.repetition_score_fail)),
        )

        # Resources
        res_raw = raw.get("resources", {})
        config.resources = ResourceConfig(
            batch_size=str(res_raw.get("batch_size", config.resources.batch_size)),
            memory_headroom_percent=float(res_raw.get("memory_headroom_percent", config.resources.memory_headroom_percent)),
            per_segment_cost_gb=float(res_raw.get("per_segment_cost_gb", config.resources.per_segment_cost_gb)),
            max_batch_size=int(res_raw.get("max_batch_size", config.resources.max_batch_size)),
            min_batch_size=int(res_raw.get("min_batch_size", config.resources.min_batch_size)),
            flush_interval=int(res_raw.get("flush_interval", config.resources.flush_interval)),
            memory_pressure_threshold=float(res_raw.get("memory_pressure_threshold", config.resources.memory_pressure_threshold)),
            engine_pool_size=str(res_raw.get("engine_pool_size", config.resources.engine_pool_size)),
        )

        # NLP
        nlp_raw = raw.get("nlp", {})
        config.nlp = NLPConfig(
            spacy_model=str(nlp_raw.get("spacy_model", config.nlp.spacy_model)),
            llm_enabled=bool(nlp_raw.get("llm_enabled", config.nlp.llm_enabled)),
            llm_model=str(nlp_raw.get("llm_model", config.nlp.llm_model)),
            llm_context_window=int(nlp_raw.get("llm_context_window", config.nlp.llm_context_window)),
            llm_context_overlap=int(nlp_raw.get("llm_context_overlap", config.nlp.llm_context_overlap)),
            emotion_detection=bool(nlp_raw.get("emotion_detection", config.nlp.emotion_detection)),
            speaker_attribution=bool(nlp_raw.get("speaker_attribution", config.nlp.speaker_attribution)),
            auto_voice_traits=bool(nlp_raw.get("auto_voice_traits", config.nlp.auto_voice_traits)),
        )

        # Logging
        log_raw = raw.get("logging", {})
        config.logging = LoggingConfig(
            level=log_raw.get("level", config.logging.level),
            format=log_raw.get("format", config.logging.format),
            file_output=log_raw.get("file_output", config.logging.file_output),
            log_dir=log_raw.get("log_dir", config.logging.log_dir),
        )

        config.config_path = str(config_path)

    # Environment variable overrides
    if os.environ.get("SERVER_PORT"):
        config.server.port = int(os.environ["SERVER_PORT"])
    if os.environ.get("LOG_LEVEL"):
        config.logging.level = os.environ["LOG_LEVEL"]
    if os.environ.get("TTS_ENGINE"):
        config.tts.engine = os.environ["TTS_ENGINE"]

    return config


# Singleton
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get or create the application configuration singleton."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
