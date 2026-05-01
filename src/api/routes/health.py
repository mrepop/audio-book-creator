"""
Health Check Routes
"""

from fastapi import APIRouter
from src.api.config import get_config, get_platform_info

router = APIRouter()


@router.get("/")
async def root():
    """API information."""
    return {
        "name": "Audio Book Creator API",
        "version": "0.1.0",
        "status": "running",
    }


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    config = get_config()
    platform_info = get_platform_info()
    return {
        "status": "healthy",
        "platform": platform_info,
        "tts_engine": config.tts.engine,
    }
