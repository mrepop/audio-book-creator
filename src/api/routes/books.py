"""
Book Management Routes

Upload, parse, analyze, and manage ebooks.
"""

import uuid
import shutil
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session

from src.api.database import get_db
from src.api.schemas import (
    BookUploadResponse, BookDetailResponse, BookListResponse,
    DramatisPersonaeResponse,
)
from src.models import Book, BookFormat, Chapter

logger = logging.getLogger(__name__)

router = APIRouter()

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
BOOKS_DIR = PROJECT_ROOT / "storage" / "books"

SUPPORTED_EXTENSIONS = {".epub": BookFormat.EPUB, ".pdf": BookFormat.PDF, ".txt": BookFormat.TXT}


@router.post("/upload", response_model=BookUploadResponse)
async def upload_book(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db),
):
    """Upload an ebook file (EPUB, PDF, or TXT)."""
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported format: {ext}. Supported: {list(SUPPORTED_EXTENSIONS.keys())}")

    # Save file
    BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    file_id = str(uuid.uuid4())[:8]
    save_path = BOOKS_DIR / f"{file_id}_{file.filename}"

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    file_size = save_path.stat().st_size

    # Try to extract metadata from ebook
    title = Path(file.filename).stem.replace("_", " ").replace("-", " ").title()
    author = None
    if ext == ".epub":
        try:
            from src.ebook.epub_parser import EpubParser
            meta = EpubParser().get_metadata(str(save_path))
            if meta.get("title"):
                title = meta["title"]
            if meta.get("author"):
                author = meta["author"]
        except Exception as e:
            logger.warning(f"Could not extract EPUB metadata: {e}")

    # Create DB record
    book = Book(
        title=title,
        author=author,
        format=SUPPORTED_EXTENSIONS[ext],
        file_path=str(save_path),
        original_filename=file.filename,
        file_size_bytes=file_size,
    )
    db.add(book)
    db.commit()
    db.refresh(book)

    # Trigger background parsing
    background_tasks.add_task(_parse_book, book.id)

    logger.info(f"Book uploaded: {file.filename} (ID: {book.id})")
    return book


@router.get("/", response_model=BookListResponse)
async def list_books(db: Session = Depends(get_db)):
    """List all uploaded books."""
    books = db.query(Book).order_by(Book.created_at.desc()).all()
    return BookListResponse(books=books, total=len(books))


@router.get("/{book_id}", response_model=BookDetailResponse)
async def get_book(book_id: int, db: Session = Depends(get_db)):
    """Get detailed information about a book."""
    book = db.query(Book).filter(Book.id == book_id).first()
    if not book:
        raise HTTPException(404, "Book not found")
    return book


@router.delete("/{book_id}")
async def delete_book(book_id: int, db: Session = Depends(get_db)):
    """Delete a book and all associated data."""
    book = db.query(Book).filter(Book.id == book_id).first()
    if not book:
        raise HTTPException(404, "Book not found")

    # Delete file
    file_path = Path(book.file_path)
    if file_path.exists():
        file_path.unlink()

    db.delete(book)
    db.commit()
    return {"status": "deleted", "book_id": book_id}


@router.post("/{book_id}/parse")
async def parse_book(
    book_id: int,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db),
):
    """Trigger parsing of a book into chapters and segments."""
    book = db.query(Book).filter(Book.id == book_id).first()
    if not book:
        raise HTTPException(404, "Book not found")

    background_tasks.add_task(_parse_book, book_id)
    return {"status": "parsing_started", "book_id": book_id}


@router.post("/{book_id}/analyze")
async def analyze_book(
    book_id: int,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db),
):
    """Trigger NLP analysis (character detection, context analysis)."""
    book = db.query(Book).filter(Book.id == book_id).first()
    if not book:
        raise HTTPException(404, "Book not found")
    if not book.is_parsed:
        raise HTTPException(400, "Book must be parsed before analysis")

    background_tasks.add_task(_analyze_book, book_id)
    return {"status": "analysis_started", "book_id": book_id}


@router.get("/{book_id}/dramatis-personae", response_model=DramatisPersonaeResponse)
async def get_dramatis_personae(book_id: int, db: Session = Depends(get_db)):
    """Get the dramatis personae (character sheet) for a book."""
    book = db.query(Book).filter(Book.id == book_id).first()
    if not book:
        raise HTTPException(404, "Book not found")

    return DramatisPersonaeResponse(
        book_id=book.id,
        book_title=book.title,
        characters=book.characters,
        total_characters=len(book.characters),
    )


async def _parse_book(book_id: int):
    """Background task: parse book into chapters."""
    from src.api.database import get_db_context
    from src.ebook import parse_book as ebook_parse

    try:
        with get_db_context() as db:
            book = db.query(Book).filter(Book.id == book_id).first()
            if not book:
                return

            chapters_data = ebook_parse(book.file_path, book.format.value)

            for i, ch in enumerate(chapters_data):
                chapter = Chapter(
                    book_id=book.id,
                    number=i + 1,
                    title=ch.get("title"),
                    raw_text=ch["text"],
                    word_count=len(ch["text"].split()),
                )
                db.add(chapter)

            book.total_chapters = len(chapters_data)
            book.total_words = sum(len(ch["text"].split()) for ch in chapters_data)
            book.is_parsed = True
            logger.info(f"Book {book_id} parsed: {len(chapters_data)} chapters")
    except Exception as e:
        logger.error(f"Failed to parse book {book_id}: {e}")


async def _analyze_book(book_id: int):
    """Background task: run NLP analysis on parsed book."""
    from src.api.database import get_db_context
    from src.nlp import analyze_book as nlp_analyze

    try:
        with get_db_context() as db:
            book = db.query(Book).filter(Book.id == book_id).first()
            if not book:
                return

            nlp_analyze(db, book)
            book.is_analyzed = True
            logger.info(f"Book {book_id} NLP analysis complete")
    except Exception as e:
        logger.error(f"Failed to analyze book {book_id}: {e}")
