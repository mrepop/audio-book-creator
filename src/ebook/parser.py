"""
Ebook Parser - Format dispatcher and base class

Routes parsing to the appropriate format-specific parser.
"""

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ParsedChapter:
    """A chapter extracted from an ebook."""
    title: Optional[str]
    text: str
    number: int = 0
    metadata: Dict = field(default_factory=dict)


class EbookParser(ABC):
    """Base class for ebook format parsers."""

    @abstractmethod
    def parse(self, file_path: str) -> List[Dict]:
        """
        Parse an ebook file into chapters.

        Returns:
            List of dicts with keys: 'title', 'text', 'metadata'
        """
        pass

    @staticmethod
    def identify_dialogue(text: str) -> List[Dict]:
        """
        Split text into dialogue and narration segments.

        Returns list of segments with type and speaker attribution.
        """
        segments = []
        # Pattern matches quoted dialogue with optional attribution
        dialogue_pattern = re.compile(
            r'(?P<pre>[^"]*?)'  # Narration before dialogue
            r'"(?P<dialogue>[^"]+)"'  # Quoted dialogue
            r'(?P<post>[^"]*?)(?="|$)',  # Attribution/narration after
            re.DOTALL
        )

        last_end = 0
        for match in dialogue_pattern.finditer(text):
            # Narration before this dialogue
            pre_text = text[last_end:match.start()].strip()
            if pre_text:
                segments.append({
                    "text": pre_text,
                    "type": "narration",
                    "speaker": None,
                })

            # The dialogue itself
            dialogue_text = match.group("dialogue").strip()
            if dialogue_text:
                # Try to find speaker attribution
                speaker = _extract_speaker(text, match.start(), match.end())
                segments.append({
                    "text": dialogue_text,
                    "type": "dialogue",
                    "speaker": speaker,
                })

            last_end = match.end()

        # Remaining narration
        remaining = text[last_end:].strip()
        if remaining:
            segments.append({
                "text": remaining,
                "type": "narration",
                "speaker": None,
            })

        # If no dialogue found, return entire text as narration
        if not segments:
            segments.append({
                "text": text.strip(),
                "type": "narration",
                "speaker": None,
            })

        return segments


def _extract_speaker(text: str, dialogue_start: int, dialogue_end: int) -> Optional[str]:
    """
    Extract speaker name from dialogue attribution.
    Looks for patterns like: "Hello," said John. / John said, "Hello."
    """
    # Check after dialogue: "...", said NAME / "...", NAME said
    after = text[dialogue_end:dialogue_end + 100]
    after_patterns = [
        r'[,.]?\s*said\s+(\w+)',
        r'[,.]?\s*(\w+)\s+said',
        r'[,.]?\s*(\w+)\s+(?:replied|whispered|shouted|asked|exclaimed|muttered|murmured)',
        r'[,.]?\s*(?:replied|whispered|shouted|asked|exclaimed|muttered|murmured)\s+(\w+)',
    ]
    for pattern in after_patterns:
        m = re.search(pattern, after, re.IGNORECASE)
        if m:
            return m.group(1)

    # Check before dialogue: NAME said, "..."
    before = text[max(0, dialogue_start - 100):dialogue_start]
    before_patterns = [
        r'(\w+)\s+said[,:]?\s*$',
        r'(\w+)\s+(?:replied|whispered|shouted|asked|exclaimed|muttered|murmured)[,:]?\s*$',
    ]
    for pattern in before_patterns:
        m = re.search(pattern, before, re.IGNORECASE)
        if m:
            return m.group(1)

    return None


def parse_book(file_path: str, format: str) -> List[Dict]:
    """
    Parse an ebook file into chapters.

    Args:
        file_path: Path to the ebook file
        format: One of 'epub', 'pdf', 'txt'

    Returns:
        List of chapter dicts with 'title' and 'text' keys
    """
    from .epub_parser import EpubParser
    from .txt_parser import TxtParser

    parsers = {
        "epub": EpubParser,
        "txt": TxtParser,
    }

    parser_class = parsers.get(format)
    if not parser_class:
        raise ValueError(f"Unsupported format: {format}. Supported: {list(parsers.keys())}")

    parser = parser_class()
    chapters = parser.parse(file_path)

    logger.info(f"Parsed {len(chapters)} chapters from {Path(file_path).name}")
    return chapters
