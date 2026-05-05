"""
Character Detector

Uses spaCy NER to identify characters in text, then builds a
dramatis personae with inferred voice traits.
"""

import re
import logging
from typing import List, Dict, Optional, Set, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


# Common male first names (covers classic and modern English literature)
MALE_NAMES = {
    "adam", "albert", "alexander", "alphonse", "andrew", "anthony", "arthur",
    "basil", "benjamin", "bernard", "caleb", "cecil", "charles", "clifford",
    "daniel", "david", "edgar", "edmund", "edward", "ernest", "ethan",
    "felix", "francis", "frederick", "george", "gerald", "harold", "henry",
    "herbert", "hugh", "isaac", "jack", "jacob", "james", "john", "joseph",
    "julius", "leonard", "luke", "marcus", "mark", "matthew", "michael",
    "nathaniel", "nicholas", "oliver", "oscar", "owen", "patrick", "paul",
    "percy", "peter", "philip", "ralph", "raymond", "reginald", "richard",
    "robert", "roger", "rupert", "samuel", "sebastian", "simon", "stephen",
    "theodore", "thomas", "timothy", "victor", "vincent", "walter", "william",
}

# Common female first names
FEMALE_NAMES = {
    "abigail", "agatha", "alice", "amelia", "anne", "beatrice", "bridget",
    "caroline", "catherine", "cecilia", "charlotte", "clara", "cordelia",
    "diana", "dorothy", "edith", "eleanor", "elizabeth", "emily", "emma",
    "ethel", "florence", "frances", "gertrude", "grace", "harriet", "helen",
    "ida", "isabella", "jane", "josephine", "juliet", "justine", "katherine",
    "laura", "lucy", "mabel", "margaret", "maria", "martha", "mary",
    "matilda", "miranda", "nora", "ophelia", "penelope", "rachel",
    "rosalind", "rose", "ruth", "safie", "sarah", "sophia", "victoria",
    "violet", "vivian",
}


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

    # Words that should never be character names (checked against ALL words in name)
    STOP_NAMES = {
        # Book structure / metadata
        "project", "gutenberg", "chapter", "volume", "letter", "part",
        "contents", "preface", "introduction", "epilogue", "prologue",
        "copyright", "license", "edition", "published", "author",
        "release", "ebook", "online", "transcriber", "editor",
        "translator", "illustrator", "publisher", "printer",
        "bibliography", "appendix", "footnote", "index", "glossary",
        # Temporal (spaCy sometimes misclassifies)
        "january", "february", "march", "april", "june",
        "july", "august", "september", "october", "november", "december",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        # Religious/mythological titles often misclassified
        "god", "christ", "satan", "devil",
        # Common English words spaCy tags as PERSON
        "farewell", "creator", "paradise", "nature",
    }

    # Names that are never characters -- places, nationalities, and concepts
    # that spaCy's small model consistently mislabels as PERSON.
    KNOWN_NON_CHARACTERS = {
        # Common European places in classic literature
        "rhine", "strasburgh", "strasbourg", "arve", "matlock", "rotterdam",
        "westmorland", "montanvert", "mont blanc", "ingolstadt", "chamounix",
        "chamonix", "plainpalais", "belrive", "lausanne", "cologny",
        "evian", "thonon", "edinburgh", "oxford", "paris", "london",
        "rome", "naples", "leghorn", "windsor", "perth", "dundee",
        # Nationalities / demonyms
        "turk", "englishman", "genevan", "alpine", "archangel",
    }

    # Honorifics and forms of address to strip before name comparison
    HONORIFICS = {
        "mr", "mrs", "miss", "ms", "dr", "sir", "lady", "lord",
        "captain", "colonel", "major", "general", "professor", "reverend",
        "father", "mother", "brother", "sister", "uncle", "aunt",
        "dear", "dearest", "old", "young", "poor", "little", "good", "great",
        "beloved", "sweet", "fair", "noble", "wretched", "unhappy",
        "master", "madam", "madame", "monsieur", "mademoiselle", "signor",
        "count", "countess", "duke", "duchess", "prince", "princess",
        "king", "queen", "baron", "baroness",
    }

    # Non-person entity labels to use for filtering
    NON_PERSON_LABELS = {
        "GPE", "LOC", "ORG", "FAC", "NORP", "PRODUCT",
        "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE",
    }

    def detect_characters(self, chapters: List[Dict]) -> List[Dict]:
        """
        Detect characters in book text using chapter-aware NER.

        Args:
            chapters: list of dicts with 'number' and 'text' keys

        Returns:
            list of character dicts with:
                name, aliases, description, role, gender, age, personality,
                dialogue_count, mention_count, first_chapter
        """
        nlp = self._get_nlp()

        # Process all chapters, track entities by type and first appearance
        person_counts = Counter()
        non_person_names: Set[str] = set()
        first_chapter_map: Dict[str, int] = {}
        max_length = 1_000_000

        for chapter_info in chapters:
            ch_num = chapter_info["number"]
            text = chapter_info["text"]
            chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]

            for chunk in chunks:
                doc = nlp(chunk)
                for ent in doc.ents:
                    cleaned = self._clean_name(ent.text)
                    if not cleaned:
                        continue

                    if ent.label_ == "PERSON":
                        if len(cleaned) > 1 and not cleaned.isnumeric():
                            person_counts[cleaned] += 1
                            if cleaned not in first_chapter_map:
                                first_chapter_map[cleaned] = ch_num
                    elif ent.label_ in self.NON_PERSON_LABELS:
                        non_person_names.add(cleaned.lower())

        logger.info(
            f"Raw NER: {len(person_counts)} person entities, "
            f"{len(non_person_names)} non-person entities"
        )

        # Filter out names that also appear as non-person entities
        person_counts = self._filter_non_persons(person_counts, non_person_names)

        # Merge similar names (handles honorifics, partial matches)
        merged, all_forms_map = self._merge_similar_names(person_counts)

        # Count dialogue per character
        full_text = "\n\n".join(ch["text"] for ch in chapters)
        dialogue_counts = self._count_dialogue(full_text, merged)

        # Resolve first_chapter for merged names
        merged_first_chapter = {}
        for canonical in merged:
            chapters_seen = []
            for original_name in all_forms_map.get(canonical, [canonical]):
                if original_name in first_chapter_map:
                    chapters_seen.append(first_chapter_map[original_name])
            merged_first_chapter[canonical] = min(chapters_seen) if chapters_seen else None

        # Build character profiles, limit to top 30
        characters = []
        for name, count in merged.most_common(30):
            if count < 3:
                continue

            # Collect aliases (original forms that differ from canonical)
            aliases = [
                f for f in all_forms_map.get(name, [])
                if f != name
            ]

            char = {
                "name": name,
                "aliases": aliases,
                "dialogue_count": dialogue_counts.get(name, 0),
                "mention_count": count,
                "role": self._infer_role(count, dialogue_counts.get(name, 0), len(merged)),
                "gender": self._infer_gender(name, full_text),
                "age": None,
                "personality": None,
                "description": None,
                "first_chapter": merged_first_chapter.get(name),
            }
            characters.append(char)

        logger.info(f"Detected {len(characters)} characters (after merge + filter)")
        return characters

    # ------------------------------------------------------------------
    # Name cleaning
    # ------------------------------------------------------------------

    def _clean_name(self, raw: str) -> Optional[str]:
        """Clean NER-extracted name by stripping junk characters."""
        # Strip leading/trailing punctuation, quotes, whitespace
        name = re.sub(r'^[^a-zA-Z]+', '', raw)
        name = re.sub(r'[^a-zA-Z.\s\'-]+$', '', name)
        name = name.strip()

        if not name or len(name) < 2:
            return None

        # Must start with an uppercase letter (proper noun)
        if name[0].islower():
            return None

        # Filter short pronouns/articles
        if len(name) <= 2 and name.lower() in {
            'i', 'a', 'an', 'it', 'he', 'she', 'we', 'me', 'my', 'no', 'so',
        }:
            return None

        # Filter text fragments containing internal sentence punctuation
        # e.g. "accusations.--Poor William"
        if re.search(r'[.!?;:\u2014\u2013]', name):
            # Allow known abbreviation prefixes like "Mr." "Dr."
            if not re.match(r'^(?:Mr|Mrs|Ms|Dr|St|Jr|Sr)\.\s', name):
                # Reject if contains sentence boundary (period/excl/question + capital)
                if re.search(r'[.!?][\s\-\u2014\u2013]*[A-Z]', name):
                    return None
                # Reject em-dashes or en-dashes within the name
                if '\u2014' in name or '\u2013' in name or '--' in name:
                    return None
                # Strip trailing periods
                name = name.rstrip('.')

        if not name or len(name) < 2:
            return None

        # Check ALL words against stop names
        words = name.lower().split()
        if any(w.rstrip('.') in self.STOP_NAMES for w in words):
            return None

        return name

    # ------------------------------------------------------------------
    # Non-person entity filtering
    # ------------------------------------------------------------------

    def _filter_non_persons(self, person_counts: Counter, non_person_names: Set[str]) -> Counter:
        """Remove names that are likely places, orgs, or other non-person entities."""
        filtered = Counter()

        for name, count in person_counts.items():
            name_lower = name.lower()

            # Check against hardcoded known non-characters first
            is_known_non_char = name_lower in self.KNOWN_NON_CHARACTERS
            if not is_known_non_char:
                for word in name_lower.split():
                    if word in self.KNOWN_NON_CHARACTERS:
                        is_known_non_char = True
                        break

            if is_known_non_char:
                logger.debug(f"Filtered known non-character: {name} (count={count})")
                continue

            # Check if the FULL name matches a non-person entity from spaCy.
            # Only match on full name -- word-level matching kills real characters
            # like "Henry Clerval" when "Henry" happens to also be tagged as GPE.
            is_non_person = name_lower in non_person_names

            if is_non_person and count < 20:
                logger.debug(f"Filtered non-person entity: {name} (count={count})")
            else:
                if is_non_person:
                    logger.debug(f"Kept ambiguous entity (high count={count}): {name}")
                filtered[name] = count

        return filtered

    # ------------------------------------------------------------------
    # Name merging
    # ------------------------------------------------------------------

    def _strip_address_forms(self, name: str) -> str:
        """Strip honorifics and forms of address from a name."""
        words = name.split()
        stripped = [w for w in words if w.lower().rstrip('.') not in self.HONORIFICS]
        return " ".join(stripped) if stripped else name

    def _merge_similar_names(self, counts: Counter) -> Tuple[Counter, Dict[str, List[str]]]:
        """
        Merge counts for similar names, preferring the most complete form.

        Handles:
            - "Dearest Clerval" + "Henry Clerval" + "Clerval" -> "Henry Clerval"
            - "Victor" + "Victor Frankenstein" -> "Victor Frankenstein"
            - Does NOT merge "Victor Frankenstein" + "Alphonse Frankenstein"

        Returns:
            (merged_counts, all_forms_map)
            all_forms_map: canonical_name -> [all original name forms that were merged]
        """
        # Step 1: Group by core name (stripped of honorifics)
        core_groups: Dict[str, List[Tuple[str, int]]] = {}
        for name, count in counts.items():
            core = self._strip_address_forms(name)
            core_lower = core.lower()
            if core_lower not in core_groups:
                core_groups[core_lower] = []
            core_groups[core_lower].append((name, count))

        # Step 2: Merge groups that share significant name parts
        sorted_cores = sorted(
            core_groups.items(),
            key=lambda x: sum(c for _, c in x[1]),
            reverse=True,
        )

        merged_groups: List[List[Tuple[str, int]]] = []
        used_cores: Set[str] = set()

        for core, forms in sorted_cores:
            if core in used_cores:
                continue

            group_forms = list(forms)
            # Track ALL words across every core merged into this group.
            # This lets "Clerval" merge into the Henry+HenryClerval group
            # because the group already contains the word "clerval".
            group_words = set(core.split())

            # Iterate until no more merges found (expanding group_words each pass)
            changed = True
            while changed:
                changed = False
                for other_core, other_forms in sorted_cores:
                    if other_core == core or other_core in used_cores:
                        continue
                    other_words = set(other_core.split())

                    should_merge = False
                    shared = group_words & other_words

                    if shared and any(len(w) >= 3 for w in shared):
                        if len(other_words) == 1:
                            # Single-word candidate matches a word in the group
                            should_merge = True
                        elif other_words.issubset(group_words) or group_words.issubset(other_words):
                            # Multi-word: one must be a word-subset of the other
                            # (prevents "Alphonse Frankenstein" merging with Victor's group)
                            should_merge = True

                    if should_merge:
                        group_forms.extend(other_forms)
                        group_words.update(other_words)
                        used_cores.add(other_core)
                        changed = True

            used_cores.add(core)
            merged_groups.append(group_forms)

        # Step 3: Pick canonical name for each group
        result = Counter()
        all_forms_map: Dict[str, List[str]] = {}

        for forms in merged_groups:
            # Choose canonical: the stripped form that is longest, highest count as tiebreak
            best_name = None
            best_core_len = 0
            best_count = 0
            total_count = 0

            for orig_name, count in forms:
                stripped = self._strip_address_forms(orig_name)
                total_count += count
                if (
                    len(stripped) > best_core_len
                    or (len(stripped) == best_core_len and count > best_count)
                ):
                    best_name = stripped
                    best_core_len = len(stripped)
                    best_count = count

            if best_name:
                result[best_name] = total_count
                all_forms_map[best_name] = list({orig for orig, _ in forms})

        return result, all_forms_map

    # ------------------------------------------------------------------
    # Dialogue counting
    # ------------------------------------------------------------------

    def _count_dialogue(self, text: str, name_counts: Counter) -> Dict[str, int]:
        """
        Count dialogue lines attributed to each character.

        Uses multi-word name patterns and deduplicates across pattern types
        to avoid double-counting.
        """
        dialogue_counts: Counter = Counter()
        names = list(name_counts.keys())

        # Build lookup: name part -> canonical name
        name_lookup: Dict[str, str] = {}
        for name in names:
            name_lookup[name.lower()] = name
            for part in name.split():
                if len(part) >= 3 and part.lower() not in name_lookup:
                    name_lookup[part.lower()] = name

        speech_verbs = (
            r'(?:said|replied|asked|whispered|shouted|exclaimed|cried|muttered|'
            r'murmured|answered|declared|remarked|observed|continued|added|'
            r'called|demanded|insisted|suggested|growled|snapped|snarled|'
            r'pleaded|begged|urged|warned|sighed|groaned|laughed|sobbed|'
            r'stammered|gasped|interrupted|inquired|retorted|responded|'
            r'announced|protested|agreed|acknowledged|confessed|admitted)'
        )

        # Capture multi-word proper names (1-3 capitalized words)
        name_pat = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})'

        patterns = [
            # "..." said Victor Frankenstein
            re.compile(
                r'["\u201c\u2018][^"\u201d\u201c\u2019]+["\u201d\u2019]'
                r'\s*[,.]?\s*' + speech_verbs + r'\s+' + name_pat
            ),
            # "..." Victor Frankenstein said
            re.compile(
                r'["\u201c\u2018][^"\u201d\u201c\u2019]+["\u201d\u2019]'
                r'\s*[,.]?\s*' + name_pat + r'\s+' + speech_verbs
            ),
            # Victor Frankenstein said, "..."
            re.compile(
                name_pat + r'\s+' + speech_verbs + r'\s*[,:]?\s*["\u201c\u2018]'
            ),
        ]

        # Track match positions to avoid double-counting the same dialogue
        attributed_positions: Set[int] = set()

        for pattern in patterns:
            for match in pattern.finditer(text):
                quote_pos = match.start()
                if quote_pos in attributed_positions:
                    continue

                speaker_text = match.group(1).strip()
                canonical = self._resolve_speaker(speaker_text, name_lookup)
                if canonical:
                    dialogue_counts[canonical] += 1
                    attributed_positions.add(quote_pos)

        return dict(dialogue_counts)

    def _resolve_speaker(self, speaker_text: str, name_lookup: Dict[str, str]) -> Optional[str]:
        """Resolve a speaker reference to a canonical character name."""
        # Strip any honorifics from the speaker reference
        stripped = self._strip_address_forms(speaker_text)

        # Try full name match
        if stripped.lower() in name_lookup:
            return name_lookup[stripped.lower()]

        # Try individual words (for partial matches like "Victor" -> "Victor Frankenstein")
        for word in stripped.split():
            if len(word) >= 3 and word.lower() in name_lookup:
                return name_lookup[word.lower()]

        return None

    # ------------------------------------------------------------------
    # Role and gender inference
    # ------------------------------------------------------------------

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
        """
        Infer gender using name database first, then pronoun proximity.

        Returns 'male', 'female', or None if ambiguous.
        """
        # Primary: check common name database
        first_name = name.split()[0].lower()
        if first_name in MALE_NAMES:
            return "male"
        if first_name in FEMALE_NAMES:
            return "female"

        # Secondary: pronoun proximity analysis (wider window, more text)
        escaped = re.escape(name)
        male_patterns = [
            rf'{escaped}.{{0,300}}\b(?:he|him|his)\b',
            rf'\b(?:he|him|his)\b.{{0,300}}{escaped}',
        ]
        female_patterns = [
            rf'{escaped}.{{0,300}}\b(?:she|her|hers)\b',
            rf'\b(?:she|her|hers)\b.{{0,300}}{escaped}',
        ]

        # Search first 500k chars (more coverage than before)
        search_text = text[:500_000]
        male_count = sum(
            len(re.findall(p, search_text, re.IGNORECASE | re.DOTALL))
            for p in male_patterns
        )
        female_count = sum(
            len(re.findall(p, search_text, re.IGNORECASE | re.DOTALL))
            for p in female_patterns
        )

        # Need at least 2 hits total and 60% confidence
        total = male_count + female_count
        if total >= 2:
            if male_count / total > 0.6:
                return "male"
            elif female_count / total > 0.6:
                return "female"

        return None
