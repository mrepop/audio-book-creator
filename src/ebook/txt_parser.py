"""
TXT Parser

Parses plain text files, detecting chapter boundaries via common patterns.
"""

import logging
import re
from typing import List, Dict

from .parser import EbookParser

logger = logging.getLogger(__name__)


class TxtParser(EbookParser):
    """Parser for plain text ebooks."""

    # Common chapter heading patterns
    CHAPTER_PATTERNS = [
        re.compile(r'^(?:Chapter|CHAPTER)\s+(?:\d+|[IVXLC]+)(?:\s*[:\-\.]\s*(.+))?$', re.MULTILINE),
        re.compile(r'^(?:Part|PART)\s+(?:\d+|[IVXLC]+)(?:\s*[:\-\.]\s*(.+))?$', re.MULTILINE),
        re.compile(r'^\d+\.\s+(.+)$', re.MULTILINE),
        re.compile(r'^={3,}$', re.MULTILINE),  # Section break
        re.compile(r'^\*{3,}$', re.MULTILINE),  # Scene break
    ]

    def parse(self, file_path: str) -> List[Dict]:
        """Parse TXT file into chapters."""
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        # Find chapter boundaries
        boundaries = self._find_chapter_boundaries(text)

        if not boundaries:
            # No chapters detected - treat as single chapter
            return [{
                "title": None,
                "text": text.strip(),
                "metadata": {},
            }]

        chapters = []
        for i, (start, title) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
            chapter_text = text[start:end].strip()

            # Remove the chapter heading from the text body
            if title:
                chapter_text = chapter_text[len(title):].strip()

            if chapter_text and len(chapter_text) > 20:
                chapters.append({
                    "title": title,
                    "text": chapter_text,
                    "metadata": {},
                })

        logger.info(f"Extracted {len(chapters)} chapters from TXT")
        return chapters

    def _find_chapter_boundaries(self, text: str) -> List[tuple]:
        """Find chapter start positions and titles in text."""
        boundaries = []

        for pattern in self.CHAPTER_PATTERNS:
            for match in pattern.finditer(text):
                title = match.group(0).strip()
                boundaries.append((match.start(), title))

        # Sort by position and deduplicate nearby boundaries
        boundaries.sort(key=lambda x: x[0])
        filtered = []
        for b in boundaries:
            if not filtered or (b[0] - filtered[-1][0]) > 200:
                filtered.append(b)

        return filtered
