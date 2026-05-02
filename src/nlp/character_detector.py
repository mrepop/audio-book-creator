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

    # Words that should never be character names
    STOP_NAMES = {
        "project", "gutenberg", "chapter", "volume", "letter", "part",
        "contents", "preface", "introduction", "epilogue", "prologue",
        "copyright", "license", "edition", "published", "author",
    }

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
                    name = self._clean_name(ent.text)
                    if name and len(name) > 1 and not name.isnumeric():
                        person_counts[name] += 1

        # Merge similar names (e.g., "John" and "John Smith")
        merged = self._merge_similar_names(person_counts)

        # Count dialogue per character
        dialogue_counts = self._count_dialogue(text, merged)

        # Build character profiles, limit to top 30 real characters
        characters = []
        for name, count in merged.most_common(30):
            if count < 3:
                continue  # Skip low-mention names

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

    def _clean_name(self, raw: str) -> Optional[str]:
        """Clean NER-extracted name by stripping junk characters."""
        # Strip leading/trailing punctuation, quotes, whitespace
        name = re.sub(r'^[^a-zA-Z]+', '', raw)
        name = re.sub(r'[^a-zA-Z.\s]+$', '', name)
        name = name.strip()

        if not name or len(name) < 2:
            return None

        # Filter out stop words and non-character names
        if name.lower().split()[0] in self.STOP_NAMES:
            return None

        # Filter names that are too short after cleaning
        # (single letters, common words)
        if len(name) <= 2 and name.lower() in {'i', 'a', 'an', 'it', 'he', 'she', 'we', 'me'}:
            return None

        return name

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

        speech_verbs = r'(?:said|replied|asked|whispered|shouted|exclaimed|cried|muttered|murmured|answered|declared|remarked|observed|continued|added)'

        # Handle multiple quote styles: straight, curly, and guillemets
        quote_patterns = [
            re.compile(r'["\u201c\u201d][^"\u201c\u201d]+["\u201c\u201d][^"\u201c\u201d]*?(\w+)\s+' + speech_verbs, re.IGNORECASE),
            re.compile(r'(\w+)\s+' + speech_verbs + r'[,:]?\s*["\u201c]', re.IGNORECASE),
        ]

        # Also match patterns like: NAME said, "..."
        # and "..." said NAME
        for pattern in quote_patterns:
            for match in pattern.finditer(text):
                speaker = match.group(1)
                for name in names:
                    name_parts = name.lower().split()
                    if speaker.lower() in name_parts or name.lower().startswith(speaker.lower()):
                        dialogue_counts[name] = dialogue_counts.get(name, 0) + 1
                        break

        # Also count by proximity: name appears within 200 chars of a speech verb
        for name in names:
            nearby_pattern = re.compile(
                rf'\b{re.escape(name)}\b.{{0,100}}\b{speech_verbs}\b|\b{speech_verbs}\b.{{0,100}}\b{re.escape(name)}\b',
                re.IGNORECASE
            )
            proximity_count = len(nearby_pattern.findall(text))
            if proximity_count > 0:
                dialogue_counts[name] = dialogue_counts.get(name, 0) + proximity_count

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
        escaped = re.escape(name)
        # Wider window: pronouns within 200 chars of the name
        male_patterns = [
            rf'{escaped}.{{0,200}}\b(?:he|him|his)\b',
            rf'\b(?:he|him|his)\b.{{0,200}}{escaped}',
        ]
        female_patterns = [
            rf'{escaped}.{{0,200}}\b(?:she|her|hers)\b',
            rf'\b(?:she|her|hers)\b.{{0,200}}{escaped}',
        ]

        # Search more of the text (first 200k chars)
        search_text = text[:200000]
        male_count = sum(len(re.findall(p, search_text, re.IGNORECASE | re.DOTALL)) for p in male_patterns)
        female_count = sum(len(re.findall(p, search_text, re.IGNORECASE | re.DOTALL)) for p in female_patterns)

        if male_count > female_count and male_count >= 2:
            return "male"
        elif female_count > male_count and female_count >= 2:
            return "female"
        return None
