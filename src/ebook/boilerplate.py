"""
Boilerplate Detection for Ebook Content

Identifies chapters and text that are metadata, table of contents,
copyright notices, or other non-narrative content that should not
be converted to speech.

Detects:
- Project Gutenberg header/footer
- Table of contents
- Copyright/license notices
- Title pages and metadata blocks
- Appendices and indices
"""

import re
import logging

logger = logging.getLogger(__name__)

# Patterns that indicate a chapter is boilerplate (not story content)
BOILERPLATE_CHAPTER_PATTERNS = [
    r"(?i)^\s*contents?\s*$",
    r"(?i)^\s*table\s+of\s+contents?\s*$",
    r"(?i)^\s*copyright",
    r"(?i)^\s*license",
    r"(?i)^\s*preface\s*$",
    r"(?i)^\s*appendix",
    r"(?i)^\s*index\s*$",
    r"(?i)^\s*glossary\s*$",
    r"(?i)^\s*bibliography\s*$",
    r"(?i)^\s*acknowledgements?\s*$",
    r"(?i)^\s*about\s+the\s+author\s*$",
    r"(?i)^\s*colophon\s*$",
]

# Patterns in text body that indicate boilerplate
BOILERPLATE_TEXT_PATTERNS = [
    r"(?i)project\s+gutenberg",
    r"(?i)this\s+ebook\s+is\s+for\s+the\s+use\s+of\s+anyone",
    r"(?i)terms\s+of\s+the\s+project\s+gutenberg\s+license",
    r"(?i)\*\*\*\s*START\s+OF\s+(THE\s+)?PROJECT\s+GUTENBERG",
    r"(?i)\*\*\*\s*END\s+OF\s+(THE\s+)?PROJECT\s+GUTENBERG",
    r"(?i)full\s+license\s+at\s+.*gutenberg\.org",
    r"(?i)www\.gutenberg\.org",
    r"(?i)most\s+recently\s+updated",
    r"(?i)release\s+date\s*:",
    r"(?i)language\s*:\s*english",
    r"(?i)credits?\s*:",
    r"(?i)html\s+version\s+by",
    r"(?i)produced\s+by",
    r"(?i)transcriber.?s?\s+note",
]

# Chapter titles that are likely just metadata listings
METADATA_TITLE_PATTERNS = [
    r"(?i)^frankenstein;\s*$",  # Title page fragments
    r"(?i)^or,?\s+the\s+modern\s+prometheus\s*$",
    r"(?i)^by\s+",  # "by Author Name"
]


def is_boilerplate_chapter(title: str, text: str) -> bool:
    """Detect if a chapter is boilerplate (not story content).

    Args:
        title: Chapter title (may be None).
        text: First ~2000 chars of chapter text.

    Returns:
        True if the chapter appears to be boilerplate.
    """
    # Check title patterns
    if title:
        for pattern in BOILERPLATE_CHAPTER_PATTERNS:
            if re.search(pattern, title):
                logger.debug(f"Boilerplate chapter detected by title: '{title}'")
                return True

    # Check text content (first 2000 chars)
    sample = text[:2000] if text else ""

    # Count how many boilerplate patterns match
    matches = 0
    for pattern in BOILERPLATE_TEXT_PATTERNS:
        if re.search(pattern, sample):
            matches += 1

    # If 2+ patterns match, it's likely boilerplate
    if matches >= 2:
        logger.debug(f"Boilerplate chapter detected: {matches} text patterns matched")
        return True

    # Check if the text is mostly a list (TOC-like)
    lines = [l.strip() for l in sample.split("\n") if l.strip()]
    if len(lines) > 5:
        short_lines = sum(1 for l in lines if len(l) < 40)
        if short_lines / len(lines) > 0.7:
            # Most lines are short -- likely TOC or list
            logger.debug(f"Boilerplate chapter detected: {short_lines}/{len(lines)} short lines (TOC-like)")
            return True

    return False


def is_boilerplate_segment(text: str) -> bool:
    """Detect if a segment is boilerplate text that shouldn't be spoken.

    Args:
        text: Segment text.

    Returns:
        True if the segment is boilerplate.
    """
    text = text.strip()

    # Very short metadata-like text
    for pattern in BOILERPLATE_TEXT_PATTERNS:
        if re.search(pattern, text):
            return True

    for pattern in METADATA_TITLE_PATTERNS:
        if re.search(pattern, text):
            return True

    return False
