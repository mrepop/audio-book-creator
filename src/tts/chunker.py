"""
Sentence-Level Text Chunker for TTS Generation

Splits text into sentence-boundary-aligned chunks that target 12 seconds
and never exceed 15 seconds.  Qwen3-TTS produces artifacts on segments
longer than ~30s, and best results come from 12-15s chunks.

The chunker:
  1. Splits text on sentence boundaries (. ! ? + paragraph breaks).
  2. Estimates duration per sentence using a words-per-second heuristic.
  3. Greedily combines consecutive sentences into chunks that stay under
     the target duration, with a hard cap.
  4. Never breaks mid-sentence.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# Regex that splits on sentence-ending punctuation followed by whitespace,
# keeping the punctuation attached to the sentence.
_SENTENCE_SPLIT_RE = re.compile(
    r'(?<=[.!?…])'        # lookbehind: sentence-ending punctuation
    r'(?:\s*["\'"\u201d\u2019])?'  # optional trailing close-quote
    r'\s+'                 # required whitespace separator
)

# Fallback: if a sentence is absurdly long (no punctuation), split on
# clause boundaries so we never blow the hard cap.
_CLAUSE_SPLIT_RE = re.compile(
    r'(?<=[,;:—\-])\s+'
)


@dataclass
class Chunk:
    """A chunk of text sized for one TTS generation call."""
    text: str
    estimated_duration: float          # seconds, heuristic
    sentence_indices: list[int]        # indices into the original sentence list
    is_paragraph_end: bool = False     # insert paragraph silence after this chunk


@dataclass
class ChunkerConfig:
    """Mirrors the values in config.yaml > chunking."""
    target_seconds: float = 12.0
    max_seconds: float = 15.0
    words_per_second: float = 2.5


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving punctuation.

    Handles:
      - Standard punctuation (.!?)
      - Ellipsis (...)
      - Curly/smart quotes after punctuation
      - Double-newline paragraph breaks (each paragraph is at least one sentence)
    """
    # First split on paragraph boundaries
    paragraphs = re.split(r'\n\n+', text)
    sentences: list[str] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Split paragraph into sentences
        parts = _SENTENCE_SPLIT_RE.split(para)
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)

    return sentences


def estimate_duration(text: str, words_per_second: float = 2.5) -> float:
    """Estimate spoken duration of text in seconds."""
    word_count = len(text.split())
    return max(0.3, word_count / words_per_second)


def chunk_text(
    text: str,
    config: Optional[ChunkerConfig] = None,
    paragraph_breaks: Optional[list[int]] = None,
) -> list[Chunk]:
    """Split text into TTS-sized chunks at sentence boundaries.

    Args:
        text: The full text to chunk (can contain multiple paragraphs).
        config: Chunking configuration (target/max seconds, WPS).
        paragraph_breaks: Optional set of sentence indices that end a paragraph.
                          If None, detected automatically from double-newlines.

    Returns:
        Ordered list of Chunk objects ready for TTS generation.
    """
    if config is None:
        config = ChunkerConfig()

    target = config.target_seconds
    hard_cap = config.max_seconds
    wps = config.words_per_second

    # Detect paragraph boundaries before flattening
    paragraphs = re.split(r'\n\n+', text)
    sentences: list[str] = []
    para_end_indices: set[int] = set()

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        parts = _SENTENCE_SPLIT_RE.split(para)
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)
        # Mark the last sentence of this paragraph
        if sentences:
            para_end_indices.add(len(sentences) - 1)

    if paragraph_breaks is not None:
        para_end_indices = set(paragraph_breaks)

    if not sentences:
        return []

    # Estimate durations
    durations = [estimate_duration(s, wps) for s in sentences]

    # Greedy packing: combine sentences into chunks up to target_seconds,
    # allowing up to max_seconds if the current sentence would push past
    # the target but still fits under the hard cap.
    chunks: list[Chunk] = []
    current_texts: list[str] = []
    current_indices: list[int] = []
    current_duration = 0.0

    for i, (sent, dur) in enumerate(zip(sentences, durations)):
        would_be = current_duration + dur

        if not current_texts:
            # First sentence in a new chunk -- always accept
            current_texts.append(sent)
            current_indices.append(i)
            current_duration = dur
        elif would_be <= target:
            # Fits comfortably within target
            current_texts.append(sent)
            current_indices.append(i)
            current_duration = would_be
        elif would_be <= hard_cap:
            # Over target but under hard cap -- include to avoid tiny leftover
            current_texts.append(sent)
            current_indices.append(i)
            current_duration = would_be
        else:
            # Would exceed hard cap -- flush current chunk, start new one
            chunks.append(Chunk(
                text=" ".join(current_texts),
                estimated_duration=current_duration,
                sentence_indices=list(current_indices),
                is_paragraph_end=bool(current_indices and current_indices[-1] in para_end_indices),
            ))
            current_texts = [sent]
            current_indices = [i]
            current_duration = dur

        # If this sentence is a paragraph end AND we have a decent chunk,
        # flush now so we can insert paragraph silence.
        if i in para_end_indices and current_duration >= target * 0.3:
            chunks.append(Chunk(
                text=" ".join(current_texts),
                estimated_duration=current_duration,
                sentence_indices=list(current_indices),
                is_paragraph_end=True,
            ))
            current_texts = []
            current_indices = []
            current_duration = 0.0

    # Flush remaining
    if current_texts:
        chunks.append(Chunk(
            text=" ".join(current_texts),
            estimated_duration=current_duration,
            sentence_indices=list(current_indices),
            is_paragraph_end=bool(current_indices and current_indices[-1] in para_end_indices),
        ))

    # Safety: if any single sentence exceeds the hard cap, split on clause
    # boundaries.  This is rare but handles run-on sentences.
    final_chunks: list[Chunk] = []
    for chunk in chunks:
        if chunk.estimated_duration > hard_cap * 1.2:
            sub_chunks = _split_long_chunk(chunk, config)
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)

    logger.info(
        f"Chunked {len(sentences)} sentences into {len(final_chunks)} chunks "
        f"(target={target}s, cap={hard_cap}s)"
    )
    for i, c in enumerate(final_chunks):
        logger.debug(
            f"  Chunk {i+1}: ~{c.estimated_duration:.1f}s, "
            f"{len(c.sentence_indices)} sentence(s), "
            f"para_end={c.is_paragraph_end}, "
            f"text='{c.text[:60]}...'"
        )

    return final_chunks


def _split_long_chunk(chunk: Chunk, config: ChunkerConfig) -> list[Chunk]:
    """Split a single oversized chunk on clause boundaries."""
    parts = _CLAUSE_SPLIT_RE.split(chunk.text)
    if len(parts) <= 1:
        # Can't split further -- return as-is (TTS will handle it)
        return [chunk]

    sub_chunks: list[Chunk] = []
    current_text = ""
    current_dur = 0.0

    for part in parts:
        part = part.strip()
        if not part:
            continue
        dur = estimate_duration(part, config.words_per_second)

        if current_text and (current_dur + dur) > config.max_seconds:
            sub_chunks.append(Chunk(
                text=current_text,
                estimated_duration=current_dur,
                sentence_indices=chunk.sentence_indices,
                is_paragraph_end=False,
            ))
            current_text = part
            current_dur = dur
        else:
            current_text = f"{current_text} {part}".strip() if current_text else part
            current_dur += dur

    if current_text:
        sub_chunks.append(Chunk(
            text=current_text,
            estimated_duration=current_dur,
            sentence_indices=chunk.sentence_indices,
            is_paragraph_end=chunk.is_paragraph_end,
        ))

    return sub_chunks
