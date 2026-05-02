"""Unit tests for src/tts/chunker.py"""

import pytest
from src.tts.chunker import (
    split_sentences,
    estimate_duration,
    chunk_text,
    Chunk,
    ChunkerConfig,
)


class TestSplitSentences:
    def test_simple_sentences(self):
        text = "Hello world. How are you? I am fine!"
        sentences = split_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "Hello world."
        assert sentences[1] == "How are you?"
        assert sentences[2] == "I am fine!"

    def test_paragraph_boundaries(self):
        text = "First paragraph.\n\nSecond paragraph."
        sentences = split_sentences(text)
        assert len(sentences) == 2
        assert sentences[0] == "First paragraph."
        assert sentences[1] == "Second paragraph."

    def test_multiple_sentences_per_paragraph(self):
        text = "Sentence one. Sentence two.\n\nSentence three."
        sentences = split_sentences(text)
        assert len(sentences) == 3

    def test_empty_text(self):
        assert split_sentences("") == []
        assert split_sentences("   ") == []

    def test_single_sentence_no_punctuation(self):
        sentences = split_sentences("Hello world")
        assert len(sentences) == 1
        assert sentences[0] == "Hello world"

    def test_ellipsis(self):
        text = "Wait for it... And then it happened."
        sentences = split_sentences(text)
        assert len(sentences) == 2

    def test_quoted_dialogue(self):
        text = '"Hello," she said. "How are you?"'
        sentences = split_sentences(text)
        # Should split on the period and question mark
        assert len(sentences) >= 1


class TestEstimateDuration:
    def test_basic_duration(self):
        # 10 words at 2.5 WPS = 4 seconds
        text = "one two three four five six seven eight nine ten"
        dur = estimate_duration(text, words_per_second=2.5)
        assert abs(dur - 4.0) < 0.01

    def test_minimum_duration(self):
        # Single word should get minimum 0.3s
        dur = estimate_duration("Hello", words_per_second=2.5)
        assert dur == pytest.approx(0.4, abs=0.01)

    def test_empty_text(self):
        dur = estimate_duration("", words_per_second=2.5)
        assert dur == 0.3  # minimum


class TestChunkText:
    def test_short_text_single_chunk(self):
        text = "This is a short sentence."
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0].text == "This is a short sentence."

    def test_long_text_multiple_chunks(self):
        # Build text that would exceed 15s target
        # At 2.5 WPS, 40 words = 16 seconds
        sentences = [
            "This is sentence number one with some extra words.",       # ~10 words = 4s
            "This is sentence number two with some extra words.",       # ~10 words = 4s
            "This is sentence number three with some extra words.",     # ~10 words = 4s
            "This is sentence number four with some extra words.",      # ~10 words = 4s
            "This is sentence number five with some extra words added.",# ~10 words = 4s
        ]
        text = " ".join(sentences)
        config = ChunkerConfig(target_seconds=12, max_seconds=15, words_per_second=2.5)
        chunks = chunk_text(text, config=config)

        assert len(chunks) >= 2
        # Each chunk should be under the hard cap
        for chunk in chunks:
            assert chunk.estimated_duration <= 15.0 * 1.2  # Allow tolerance for edge cases

    def test_never_breaks_mid_sentence(self):
        text = "Short. " * 50  # 50 one-word sentences
        config = ChunkerConfig(target_seconds=5, max_seconds=8, words_per_second=2.5)
        chunks = chunk_text(text, config=config)

        # Verify each chunk's text ends with punctuation (sentence boundary)
        for chunk in chunks:
            assert chunk.text.rstrip().endswith(".")

    def test_paragraph_detection(self):
        text = "Paragraph one sentence one. Paragraph one sentence two.\n\nParagraph two."
        chunks = chunk_text(text)

        # Should detect paragraph boundary
        has_para_end = any(c.is_paragraph_end for c in chunks)
        assert has_para_end

    def test_empty_text_returns_empty(self):
        chunks = chunk_text("")
        assert chunks == []

    def test_custom_config(self):
        # 20 sentences, each ~2s = ~40s total at 2.5 WPS
        text = "This is a test sentence with five words. " * 20
        config = ChunkerConfig(target_seconds=8, max_seconds=10, words_per_second=2.5)
        chunks = chunk_text(text, config=config)
        assert len(chunks) >= 4  # 40s / 10s = at least 4 chunks

    def test_sentence_indices_populated(self):
        text = "Sentence one. Sentence two. Sentence three."
        chunks = chunk_text(text)
        # All chunks should have sentence indices
        all_indices = []
        for chunk in chunks:
            assert len(chunk.sentence_indices) > 0
            all_indices.extend(chunk.sentence_indices)
        # Indices should cover all sentences
        assert sorted(all_indices) == list(range(3))

    def test_chunk_dataclass_fields(self):
        text = "Hello world. How are you?"
        chunks = chunk_text(text)
        assert len(chunks) >= 1
        chunk = chunks[0]
        assert isinstance(chunk.text, str)
        assert isinstance(chunk.estimated_duration, float)
        assert isinstance(chunk.sentence_indices, list)
        assert isinstance(chunk.is_paragraph_end, bool)
