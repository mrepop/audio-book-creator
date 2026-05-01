"""
Dialogue Parser

Splits chapter text into dialogue and narration segments with
speaker attribution for TTS voice assignment.
"""

import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class DialogueParser:
    """Parses text into dialogue and narration segments."""

    def parse_chapter(self, text: str) -> List[Dict]:
        """
        Parse chapter text into ordered segments.

        Returns list of dicts with keys:
            text, type ('dialogue' or 'narration'), speaker (name or None)
        """
        segments = []
        # Split into paragraphs first
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

        for para in paragraphs:
            para_segments = self._parse_paragraph(para)
            segments.extend(para_segments)

        # Merge adjacent narration segments that are very short
        merged = self._merge_short_segments(segments)

        return merged

    def _parse_paragraph(self, paragraph: str) -> List[Dict]:
        """Parse a single paragraph into segments."""
        segments = []
        # Find all quoted dialogue
        pattern = re.compile(r'"([^"]+)"')

        last_end = 0
        for match in pattern.finditer(paragraph):
            # Narration before dialogue
            pre = paragraph[last_end:match.start()].strip()
            if pre:
                segments.append({
                    "text": pre,
                    "type": "narration",
                    "speaker": None,
                })

            # Dialogue
            dialogue_text = match.group(1).strip()
            if dialogue_text:
                speaker = self._find_speaker(paragraph, match.start(), match.end())
                segments.append({
                    "text": dialogue_text,
                    "type": "dialogue",
                    "speaker": speaker,
                })

            last_end = match.end()

        # Remaining text after last dialogue
        remaining = paragraph[last_end:].strip()
        if remaining:
            segments.append({
                "text": remaining,
                "type": "narration",
                "speaker": None,
            })

        # If no segments created (no dialogue), treat whole paragraph as narration
        if not segments:
            segments.append({
                "text": paragraph,
                "type": "narration",
                "speaker": None,
            })

        return segments

    def _find_speaker(self, text: str, dialog_start: int, dialog_end: int) -> Optional[str]:
        """Extract speaker from dialogue attribution."""
        speech_verbs = (
            "said|replied|asked|whispered|shouted|exclaimed|muttered|"
            "murmured|cried|called|answered|demanded|insisted|suggested|"
            "declared|announced|remarked|observed|noted|continued|added|"
            "growled|snapped|snarled|pleaded|begged|urged|warned|"
            "sighed|groaned|laughed|chuckled|giggled|sobbed"
        )

        # Check after dialogue
        after = text[dialog_end:dialog_end + 150]
        after_match = re.search(
            rf'[,.]?\s*(?:({speech_verbs})\s+(\w+)|(\w+)\s+({speech_verbs}))',
            after, re.IGNORECASE
        )
        if after_match:
            return after_match.group(2) or after_match.group(3)

        # Check before dialogue
        before = text[max(0, dialog_start - 150):dialog_start]
        before_match = re.search(
            rf'(\w+)\s+(?:{speech_verbs})[,:]?\s*$',
            before, re.IGNORECASE
        )
        if before_match:
            return before_match.group(1)

        return None

    def _merge_short_segments(self, segments: List[Dict], min_words: int = 3) -> List[Dict]:
        """Merge very short narration segments with adjacent ones."""
        if len(segments) <= 1:
            return segments

        merged = [segments[0]]
        for seg in segments[1:]:
            prev = merged[-1]
            # Merge short narration into previous narration
            if (
                seg["type"] == "narration"
                and prev["type"] == "narration"
                and len(seg["text"].split()) < min_words
            ):
                prev["text"] = prev["text"] + " " + seg["text"]
            else:
                merged.append(seg)

        return merged
