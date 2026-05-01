"""
EPUB Parser

Extracts chapters and metadata from EPUB files using ebooklib.
"""

import logging
import re
from typing import List, Dict, Optional

from bs4 import BeautifulSoup

from .parser import EbookParser

logger = logging.getLogger(__name__)


class EpubParser(EbookParser):
    """Parser for EPUB format ebooks."""

    def parse(self, file_path: str) -> List[Dict]:
        """Parse EPUB file into chapters."""
        import ebooklib
        from ebooklib import epub

        book = epub.read_epub(file_path)
        chapters = []

        # Extract metadata
        title = book.get_metadata("DC", "title")
        author = book.get_metadata("DC", "creator")

        logger.info(f"Parsing EPUB: {title[0][0] if title else 'Unknown'}")

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
                },
            })

        logger.info(f"Extracted {len(chapters)} chapters from EPUB")
        return chapters

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract clean text from HTML, preserving paragraph structure."""
        # Remove script and style elements
        for tag in soup(["script", "style", "nav"]):
            tag.decompose()

        # Get text with paragraph breaks
        paragraphs = []
        for p in soup.find_all(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6"]):
            text = p.get_text(strip=True)
            if text:
                paragraphs.append(text)

        # If no paragraphs found, get all text
        if not paragraphs:
            return soup.get_text(separator="\n", strip=True)

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
