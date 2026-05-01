"""
Character Management Routes
"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.database import get_db
from src.api.schemas import CharacterResponse, CharacterUpdateRequest
from src.models import Character

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/book/{book_id}", response_model=list[CharacterResponse])
async def list_characters(book_id: int, db: Session = Depends(get_db)):
    """List all characters for a book."""
    characters = (
        db.query(Character)
        .filter(Character.book_id == book_id)
        .order_by(Character.dialogue_count.desc())
        .all()
    )
    return characters


@router.get("/{character_id}", response_model=CharacterResponse)
async def get_character(character_id: int, db: Session = Depends(get_db)):
    """Get character details."""
    character = db.query(Character).filter(Character.id == character_id).first()
    if not character:
        raise HTTPException(404, "Character not found")
    return character


@router.patch("/{character_id}", response_model=CharacterResponse)
async def update_character(
    character_id: int,
    update: CharacterUpdateRequest,
    db: Session = Depends(get_db),
):
    """Update character details (name, description, role, voice profile)."""
    character = db.query(Character).filter(Character.id == character_id).first()
    if not character:
        raise HTTPException(404, "Character not found")

    for field, value in update.model_dump(exclude_unset=True).items():
        setattr(character, field, value)

    db.commit()
    db.refresh(character)
    return character
