"""
Audio Book Creator API

FastAPI application for converting ebooks into audiobooks
with realistic, context-aware voices using Qwen3 TTS.
"""

import sys
import os
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.database import init_db
from src.api.config import get_config, get_platform_info
from src.api.routes import books, characters, voices, generation, health

# Load configuration
config = get_config()
platform_info = get_platform_info()

# ---- Logging setup: console + rotating file ----
LOG_DIR = PROJECT_ROOT / config.logging.log_dir
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_level = getattr(logging, config.logging.level.upper(), logging.INFO)
log_format = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
date_format = "%Y-%m-%d %H:%M:%S"
formatter = logging.Formatter(log_format, datefmt=date_format)

# Root logger -- clear any existing handlers first (prevents duplicates on reload)
root_logger = logging.getLogger()
root_logger.handlers.clear()
root_logger.setLevel(log_level)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(log_level)
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

# Main log file (rotating, 10MB, keep 10)
main_log = LOG_DIR / "audiobook.log"
file_handler = logging.handlers.RotatingFileHandler(
    main_log, maxBytes=10 * 1024 * 1024, backupCount=10, encoding="utf-8"
)
file_handler.setLevel(log_level)
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

# Error-only log file
error_log = LOG_DIR / "errors.log"
error_handler = logging.handlers.RotatingFileHandler(
    error_log, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(formatter)
root_logger.addHandler(error_handler)

# Quiet noisy third-party loggers
for noisy in ["watchfiles", "httpcore", "httpx", "urllib3", "multipart", "filelock"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ---- Startup diagnostics ----
logger.info("=" * 70)
logger.info("AUDIO BOOK CREATOR - Starting up")
logger.info("=" * 70)
logger.info(f"Config: {config.config_path}")
logger.info(f"Log level: {config.logging.level} | Log dir: {LOG_DIR}")
logger.info(f"Log files: {main_log.name}, {error_log.name}")
logger.info(f"Platform: {platform_info['os']} {platform_info['arch']}")
logger.info(f"Python: {platform_info['python_version']}")
logger.info(f"PyTorch: {platform_info.get('torch_version', 'not installed')}")
logger.info(f"GPU: {platform_info['gpu_type']} (available={platform_info['gpu_available']})")
if platform_info.get("gpu_name"):
    logger.info(f"GPU device: {platform_info['gpu_name']}")
logger.info(f"TTS engine: {config.tts.engine}")
logger.info(f"TTS model: {config.tts.model_name}")
logger.info(f"Server: {config.server.host}:{config.server.port} (reload={config.server.reload})")
logger.info("-" * 70)

# Create FastAPI app
app = FastAPI(
    title="Audio Book Creator API",
    description="Convert ebooks into high-quality audiobooks with context-aware TTS",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    import time
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    # Only log non-polling requests at DEBUG, or slow requests at INFO
    path = request.url.path
    if elapsed > 500:
        logger.warning(f"{request.method} {path} -> {response.status_code} [{elapsed:.0f}ms] SLOW")
    elif not path.startswith("/api/generation/jobs"):  # Skip polling noise
        logger.debug(f"{request.method} {path} -> {response.status_code} [{elapsed:.0f}ms]")
    return response


# Include route modules
app.include_router(health.router, tags=["Health"])
app.include_router(books.router, prefix="/api/books", tags=["Books"])
app.include_router(characters.router, prefix="/api/characters", tags=["Characters"])
app.include_router(voices.router, prefix="/api/voices", tags=["Voice Profiles"])
app.include_router(generation.router, prefix="/api/generation", tags=["Generation"])

# Serve frontend static files if they exist
frontend_dir = PROJECT_ROOT / "frontend" / "dist"
if frontend_dir.exists():
    app.mount("/app", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


@app.on_event("startup")
async def startup_event():
    """Initialize database and services on startup."""
    init_db()

    # ---- Crash recovery: mark orphaned jobs as failed ----
    from src.api.database import get_db_context
    from src.models import GenerationJob, JobStatus

    try:
        with get_db_context() as db:
            orphaned = (
                db.query(GenerationJob)
                .filter(GenerationJob.status == JobStatus.GENERATING)
                .all()
            )
            for job in orphaned:
                job.status = JobStatus.FAILED
                job.error_message = "Interrupted by server restart -- resume to continue"
                logger.warning(f"Crash recovery: job {job.job_id} marked as FAILED (was stuck in GENERATING)")
            if orphaned:
                db.commit()
                logger.info(f"Crash recovery: {len(orphaned)} orphaned job(s) marked as resumable")
    except Exception as e:
        logger.error(f"Crash recovery failed: {e}")

    logger.info("[SUCCESS] Audio Book Creator API started")
    logger.info(f"[SUCCESS] Server running on {config.server.host}:{config.server.port}")
    logger.info(f"[SUCCESS] Logs writing to {LOG_DIR}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload,
    )
