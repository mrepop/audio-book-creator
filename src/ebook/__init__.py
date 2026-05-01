"""
Ebook Parsing Module

Supports EPUB, PDF, and TXT formats. Extracts chapters, metadata,
and identifies dialogue vs narration segments.
"""

from .parser import parse_book, EbookParser
from .epub_parser import EpubParser
from .txt_parser import TxtParser

__all__ = ["parse_book", "EbookParser", "EpubParser", "TxtParser"]
