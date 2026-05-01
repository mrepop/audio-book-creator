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
import torch

logger = logging.getLogger(__name__)

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
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
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
        "torch_version": torch.__version__,
    }

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
