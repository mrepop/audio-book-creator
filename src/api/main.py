"""
Audio Book Creator API

FastAPI application for converting ebooks into audiobooks
with realistic, context-aware voices using Qwen3 TTS.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timezone

from fastapi import FastAPI
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

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.logging.level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
logger.info(f"Platform: {platform_info}")
logger.info(f"Config loaded from: {config.config_path}")

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
    logger.info("Audio Book Creator API started")
    logger.info(f"Server running on {config.server.host}:{config.server.port}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload,
    )
