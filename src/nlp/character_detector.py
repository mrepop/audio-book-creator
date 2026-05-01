"""
Character Detector

Uses spaCy NER to identify characters in text, then builds a
dramatis personae with inferred voice traits.
"""

import re
import logging
from typing import List, Dict, Optional
from collections import Counter

logger = logging.getLogger(__name__)


class CharacterDetector:
    """Detects characters in book text using NER and heuristics."""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self._nlp = None
        self._model_name = spacy_model

    def _get_nlp(self):
        """Lazy load spaCy model."""
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load(self._model_name)
                logger.info(f"spaCy model loaded: {self._model_name}")
            except OSError:
                logger.warning(f"spaCy model '{self._model_name}' not found. Run: python -m spacy download {self._model_name}")
                raise
        return self._nlp

    def detect_characters(self, text: str) -> List[Dict]:
        """
        Detect characters in the full book text.

        Returns list of character dicts with:
            name, aliases, description, role, gender, age, personality,
            dialogue_count, first_chapter
        """
        nlp = self._get_nlp()

        # Process text in chunks (spaCy has a max length)
        max_length = 1_000_000
        chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]

        # Count PERSON entities
        person_counts = Counter()
        for chunk in chunks:
            doc = nlp(chunk)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    name = ent.text.strip()
                    if len(name) > 1 and not name.isnumeric():
                        person_counts[name] += 1

        # Merge similar names (e.g., "John" and "John Smith")
        merged = self._merge_similar_names(person_counts)

        # Count dialogue per character
        dialogue_counts = self._count_dialogue(text, merged)

        # Build character profiles
        characters = []
        for name, count in merged.most_common(50):  # Top 50 characters
            if count < 2:
                continue  # Skip one-off mentions

            char = {
                "name": name,
                "aliases": [],
                "dialogue_count": dialogue_counts.get(name, 0),
                "mention_count": count,
                "role": self._infer_role(count, dialogue_counts.get(name, 0), len(merged)),
                "gender": self._infer_gender(name, text),
                "age": None,
                "personality": None,
                "description": None,
                "first_chapter": None,
            }
            characters.append(char)

        logger.info(f"Detected {len(characters)} characters")
        return characters

    def _merge_similar_names(self, counts: Counter) -> Counter:
        """Merge counts for similar names (e.g., 'John Smith' and 'John')."""
        merged = Counter()
        names = sorted(counts.keys(), key=len, reverse=True)
        used = set()

        for name in names:
            if name in used:
                continue
            total = counts[name]
            for other in names:
                if other != name and other not in used and other in name:
                    total += counts[other]
                    used.add(other)
            merged[name] = total
            used.add(name)

        return merged

    def _count_dialogue(self, text: str, name_counts: Counter) -> Dict[str, int]:
        """Count dialogue lines attributed to each character."""
        dialogue_counts = {}
        names = list(name_counts.keys())

        # Find attribution patterns near dialogue
        pattern = re.compile(r'"[^"]+"[^"]*?(\w+)\s+(?:said|replied|asked|whispered|shouted)', re.IGNORECASE)
        for match in pattern.finditer(text):
            speaker = match.group(1)
            for name in names:
                if speaker.lower() in name.lower() or name.lower().startswith(speaker.lower()):
                    dialogue_counts[name] = dialogue_counts.get(name, 0) + 1
                    break

        return dialogue_counts

    def _infer_role(self, mention_count: int, dialogue_count: int, total_chars: int) -> str:
        """Infer character role based on frequency."""
        score = mention_count + (dialogue_count * 2)
        if score > 50:
            return "protagonist"
        elif score > 20:
            return "supporting"
        else:
            return "minor"

    def _infer_gender(self, name: str, text: str) -> Optional[str]:
        """Infer gender from pronoun usage near the character name."""
        # Find pronouns used near the character name
        male_patterns = [
            rf'{name}\s+(?:he|him|his)\b',
            rf'\b(?:he|him|his)\s+{name}\b',
        ]
        female_patterns = [
            rf'{name}\s+(?:she|her|hers)\b',
            rf'\b(?:she|her|hers)\s+{name}\b',
        ]

        male_count = sum(len(re.findall(p, text[:50000], re.IGNORECASE)) for p in male_patterns)
        female_count = sum(len(re.findall(p, text[:50000], re.IGNORECASE)) for p in female_patterns)

        if male_count > female_count and male_count >= 2:
            return "male"
        elif female_count > male_count and female_count >= 2:
            return "female"
        return None
