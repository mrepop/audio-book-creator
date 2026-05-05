"""
Sentence Splitter

Splits chapter text into individual sentences using spaCy's sentence
boundary detection. Tracks paragraph membership and dialogue status
for downstream segmentation.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# Quote characters
OPEN_QUOTES = '"\u201c\u2018\u00ab'
CLOSE_QUOTES = '"\u201d\u2019\u00bb'


@dataclass
class Sentence:
    """A single sentence with metadata for context analysis."""
    text: str
    index: int                          # Global index within the chapter
    paragraph_index: int                # Which paragraph this belongs to
    is_dialogue: bool = False           # Inside quoted speech?
    is_paragraph_start: bool = False
    is_paragraph_end: bool = False
    speaker_hint: Optional[str] = None  # Nearby speaker attribution if found


def split_sentences(text: str, spacy_nlp=None) -> List[Sentence]:
    """
    Split chapter text into individual sentences.

    Uses spaCy for sentence boundary detection when available,
    falls back to regex-based splitting.

    Args:
        text: Full chapter text
        spacy_nlp: Pre-loaded spaCy model (optional, uses regex fallback if None)

    Returns:
        List of Sentence objects with paragraph and dialogue tracking
    """
    # Split into paragraphs first
    paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]

    sentences: List[Sentence] = []
    global_idx = 0

    for para_idx, para_text in enumerate(paragraphs):
        # Detect which parts of this paragraph are dialogue
        dialogue_spans = _find_dialogue_spans(para_text)

        # Split paragraph into sentences
        if spacy_nlp is not None:
            para_sents = _spacy_split(para_text, spacy_nlp)
        else:
            para_sents = _regex_split(para_text)

        for i, sent_text in enumerate(para_sents):
            if not sent_text.strip():
                continue

            # Check if this sentence falls within a dialogue span
            sent_start = para_text.find(sent_text)
            is_dlg = _in_dialogue_span(sent_start, sent_start + len(sent_text), dialogue_spans)

            sent = Sentence(
                text=sent_text.strip(),
                index=global_idx,
                paragraph_index=para_idx,
                is_dialogue=is_dlg,
                is_paragraph_start=(i == 0),
                is_paragraph_end=(i == len(para_sents) - 1),
            )
            sentences.append(sent)
            global_idx += 1

    logger.info(f"Split into {len(sentences)} sentences across {len(paragraphs)} paragraphs")
    return sentences


def _spacy_split(text: str, nlp) -> List[str]:
    """Split text using spaCy sentence boundary detection."""
    # spaCy has a max length; chunk if needed
    max_len = 1_000_000
    if len(text) <= max_len:
        doc = nlp(text)
        return [sent.text for sent in doc.sents]

    # Process in chunks
    sents = []
    for i in range(0, len(text), max_len):
        chunk = text[i:i + max_len]
        doc = nlp(chunk)
        sents.extend([sent.text for sent in doc.sents])
    return sents


def _regex_split(text: str) -> List[str]:
    """Fallback regex-based sentence splitting."""
    # Split on sentence-ending punctuation followed by space + capital letter
    # Preserves the punctuation with the sentence
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\u201c])', text)
    return [p.strip() for p in parts if p.strip()]


def _find_dialogue_spans(text: str) -> List[tuple]:
    """Find (start, end) spans of quoted dialogue in text."""
    spans = []
    pattern = re.compile(r'["\u201c]([^"\u201d\u201c]+)["\u201d]')
    for match in pattern.finditer(text):
        spans.append((match.start(), match.end()))
    return spans


def _in_dialogue_span(sent_start: int, sent_end: int, spans: List[tuple]) -> bool:
    """Check if a sentence overlaps with any dialogue span."""
    if sent_start < 0:
        return False
    for span_start, span_end in spans:
        # Sentence overlaps with dialogue if there's any intersection
        if sent_start < span_end and sent_end > span_start:
            return True
    return False
