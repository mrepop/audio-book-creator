"""
EPUB Parser

Extracts chapters and metadata from EPUB files using ebooklib.
"""

import logging
import re
import warnings
from typing import List, Dict, Optional, Tuple

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from .parser import EbookParser

logger = logging.getLogger(__name__)

# Suppress the XMLParsedAsHTMLWarning from BeautifulSoup
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


class EpubParser(EbookParser):
    """Parser for EPUB format ebooks."""

    def parse(self, file_path: str) -> List[Dict]:
        """Parse EPUB file into chapters."""
        import ebooklib
        from ebooklib import epub

        book = epub.read_epub(file_path)
        chapters = []

        # Extract metadata
        title_meta = book.get_metadata("DC", "title")
        author_meta = book.get_metadata("DC", "creator")
        book_title = title_meta[0][0] if title_meta else None
        book_author = author_meta[0][0] if author_meta else None

        logger.info(f"Parsing EPUB: {book_title or 'Unknown'} by {book_author or 'Unknown'}")

        # Process spine items (reading order)
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            content = item.get_content().decode("utf-8", errors="replace")
            soup = BeautifulSoup(content, "lxml")

            # Extract text, cleaning HTML
            text = self._extract_text(soup)
            if not text or len(text.strip()) < 50:
                continue  # Skip very short sections (ToC, copyright, etc.)

            # Try to extract chapter title
            chapter_title = self._extract_chapter_title(soup)

            chapters.append({
                "title": chapter_title,
                "text": text.strip(),
                "metadata": {
                    "item_id": item.get_id(),
                    "item_name": item.get_name(),
                    "book_title": book_title,
                    "book_author": book_author,
                },
            })

        logger.info(f"Extracted {len(chapters)} chapters from EPUB")
        return chapters

    def get_metadata(self, file_path: str) -> Dict:
        """Extract book metadata without full parsing."""
        from ebooklib import epub
        book = epub.read_epub(file_path)
        title_meta = book.get_metadata("DC", "title")
        author_meta = book.get_metadata("DC", "creator")
        lang_meta = book.get_metadata("DC", "language")
        return {
            "title": title_meta[0][0] if title_meta else None,
            "author": author_meta[0][0] if author_meta else None,
            "language": lang_meta[0][0] if lang_meta else None,
        }

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract clean text from HTML, preserving paragraph structure."""
        # Remove script and style elements
        for tag in soup(["script", "style", "nav"]):
            tag.decompose()

        # Get text with proper paragraph separation
        # Use get_text with separator to handle inline elements properly
        paragraphs = []
        for p in soup.find_all(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6"]):
            # Use separator=" " to join inline children with spaces
            text = p.get_text(separator=" ", strip=True)
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            if text:
                paragraphs.append(text)

        # If no paragraphs found, get all text
        if not paragraphs:
            raw = soup.get_text(separator=" ", strip=True)
            return re.sub(r'\s+', ' ', raw).strip()

        return "\n\n".join(paragraphs)

    def _extract_chapter_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Try to extract chapter title from HTML."""
        # Look for heading tags
        for tag_name in ["h1", "h2", "h3"]:
            tag = soup.find(tag_name)
            if tag:
                title = tag.get_text(strip=True)
                if title and len(title) < 200:
                    return title

        # Look for common chapter patterns
        text = soup.get_text()[:500]
        chapter_pattern = re.compile(
            r'(?:Chapter|CHAPTER)\s+(?:\d+|[IVXLC]+)(?:\s*[:\-\.]\s*(.+))?',
            re.IGNORECASE
        )
        match = chapter_pattern.search(text)
        if match:
            return match.group(0).strip()

        return None
