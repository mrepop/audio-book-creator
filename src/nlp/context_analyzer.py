"""
Context Analyzer

Determines emotion, emphasis, pacing, and tone for text segments
to guide TTS generation parameters.
"""

import re
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# Emotion keyword mappings
EMOTION_KEYWORDS = {
    "angry": ["angry", "furious", "rage", "shouted", "screamed", "yelled", "slammed", "cursed"],
    "sad": ["sad", "tears", "cried", "sobbed", "mourned", "grief", "wept", "sorrow"],
    "happy": ["laughed", "smiled", "grinned", "joy", "happy", "delighted", "cheerful", "beamed"],
    "fearful": ["afraid", "terrified", "trembled", "scared", "horror", "panic", "dread", "shuddered"],
    "surprised": ["gasped", "stunned", "shocked", "amazed", "startled", "astonished"],
    "tender": ["gently", "softly", "tenderly", "lovingly", "caressed", "whispered"],
    "excited": ["exclaimed", "excited", "eager", "thrilled", "enthusiasm", "rushed"],
    "contemplative": ["thought", "pondered", "considered", "reflected", "wondered", "mused"],
}

# Emphasis indicators
EMPHASIS_PATTERNS = {
    "whispered": [r"\bwhispered\b", r"\bmurmured\b", r"\bhissed\b", r"\bmuttered\b"],
    "shouted": [r"\bshouted\b", r"\byelled\b", r"\bscreamed\b", r"\bbellowed\b", r"\broared\b"],
    "soft": [r"\bsoftly\b", r"\bquietly\b", r"\bgently\b"],
    "strong": [r"\bfirmly\b", r"\bstrongly\b", r"\bforcefully\b", r"\bcommanded\b"],
}


class ContextAnalyzer:
    """Analyzes text segments for context cues that affect TTS generation."""

    def analyze_segment(self, text: str, segment_type: str = "narration") -> Dict:
        """
        Analyze a text segment for emotion, emphasis, and pacing.

        Args:
            text: The segment text
            segment_type: 'dialogue' or 'narration'

        Returns:
            Dict with keys: emotion, emphasis, pacing, context details
        """
        emotion = self._detect_emotion(text)
        emphasis = self._detect_emphasis(text)
        pacing = self._detect_pacing(text)

        return {
            "emotion": emotion,
            "emphasis": emphasis,
            "pacing": pacing,
            "segment_type": segment_type,
        }

    def _detect_emotion(self, text: str) -> Optional[str]:
        """Detect dominant emotion from text keywords."""
        text_lower = text.lower()
        scores = {}

        for emotion, keywords in EMOTION_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[emotion] = score

        if scores:
            return max(scores, key=scores.get)
        return "neutral"

    def _detect_emphasis(self, text: str) -> Optional[str]:
        """Detect emphasis style from context."""
        for emphasis, patterns in EMPHASIS_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return emphasis

        # Check for ALL CAPS words (shouting)
        caps_words = re.findall(r'\b[A-Z]{2,}\b', text)
        if len(caps_words) > 2:
            return "shouted"

        # Check for italics indicators (emphasis in text)
        if re.search(r'\*[^*]+\*|_[^_]+_', text):
            return "emphasized"

        return "normal"

    def _detect_pacing(self, text: str) -> str:
        """Detect pacing from sentence structure."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return "normal"

        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)

        # Short choppy sentences = fast/urgent
        if avg_length < 5:
            return "fast"
        # Long flowing sentences = slow/contemplative
        elif avg_length > 20:
            return "slow"
        # Exclamation marks = urgent
        elif text.count("!") > 2:
            return "urgent"
        else:
            return "normal"
