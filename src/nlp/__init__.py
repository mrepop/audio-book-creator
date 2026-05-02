"""
NLP Module - Character Detection, Context Analysis, and Dialogue Parsing

Uses spaCy for named entity recognition and custom heuristics for
dialogue attribution, emotion detection, and voice trait inference.
"""

from .character_detector import CharacterDetector
from .context_analyzer import ContextAnalyzer
from .dialogue_parser import DialogueParser


# Map from dialogue parser type strings to SegmentType enum values
_SEGMENT_TYPE_MAP = {
    "dialogue": "dialogue",
    "narration": "narration",
    "internal_thought": "internal_thought",
}


def analyze_book(db, book):
    """
    Run full NLP analysis on a parsed book.

    1. Detect characters via NER and dialogue attribution
    2. Build dramatis personae with inferred voice traits
    3. Auto-create voice profiles for detected characters
    4. Parse chapters into segments (dialogue/narration)
    5. Analyze context for each segment (emotion, emphasis, pacing)
    """
    import logging
    from src.models import Chapter, Character, Segment, SegmentType, VoiceProfile

    logger = logging.getLogger(__name__)

    detector = CharacterDetector()
    analyzer = ContextAnalyzer()
    parser = DialogueParser()

    # Collect all text for character detection
    chapters = db.query(Chapter).filter(Chapter.book_id == book.id).order_by(Chapter.number).all()
    full_text = "\n\n".join(ch.raw_text for ch in chapters)

    # Step 1: Detect characters
    logger.info(f"Detecting characters in book {book.id}...")
    characters_data = detector.detect_characters(full_text)
    character_map = {}  # name -> Character ORM object

    seed_base = 42
    for i, char_data in enumerate(characters_data):
        # Create a voice profile for this character
        profile = VoiceProfile(
            name=f"{char_data['name']}",
            description=f"Auto-generated voice for {char_data['name']}",
            gender=char_data.get("gender"),
            temperature=0.6 if char_data.get("role") == "protagonist" else 0.7,
            seed=seed_base + i,
            speaking_rate=1.0,
            pitch_shift=2.0 if char_data.get("gender") == "female" else (-2.0 if char_data.get("gender") == "male" else 0.0),
        )
        db.add(profile)
        db.flush()

        character = Character(
            book_id=book.id,
            voice_profile_id=profile.id,
            name=char_data["name"],
            aliases=str(char_data.get("aliases", [])),
            description=char_data.get("description"),
            role=char_data.get("role"),
            inferred_gender=char_data.get("gender"),
            inferred_age=char_data.get("age"),
            inferred_personality=char_data.get("personality"),
            dialogue_count=char_data.get("dialogue_count", 0),
            first_appearance_chapter=char_data.get("first_chapter"),
        )
        db.add(character)
        db.flush()
        character_map[char_data["name"].lower()] = character
        for alias in char_data.get("aliases", []):
            character_map[alias.lower()] = character

    logger.info(f"Created {len(characters_data)} characters with voice profiles")

    # Step 2: Parse each chapter into segments
    logger.info(f"Parsing {len(chapters)} chapters into segments...")
    total_segments = 0
    for chapter in chapters:
        segments_data = parser.parse_chapter(chapter.raw_text)

        for seq, seg in enumerate(segments_data):
            # Find character for dialogue segments
            character_id = None
            if seg["type"] == "dialogue" and seg.get("speaker"):
                char = character_map.get(seg["speaker"].lower())
                if char:
                    character_id = char.id

            # Analyze context
            context = analyzer.analyze_segment(seg["text"], seg["type"])

            # Map type string to enum
            seg_type_str = _SEGMENT_TYPE_MAP.get(seg["type"], "narration")
            try:
                seg_type = SegmentType(seg_type_str)
            except ValueError:
                seg_type = SegmentType.NARRATION

            segment = Segment(
                chapter_id=chapter.id,
                character_id=character_id,
                text=seg["text"],
                segment_type=seg_type,
                sequence_number=seq,
                emotion=context.get("emotion"),
                emphasis=context.get("emphasis"),
                pacing=context.get("pacing"),
            )
            db.add(segment)
            total_segments += 1

    book.total_characters = len(characters_data)
    db.flush()
    logger.info(f"Analysis complete: {len(characters_data)} characters, {total_segments} segments")


__all__ = ["CharacterDetector", "ContextAnalyzer", "DialogueParser", "analyze_book"]
