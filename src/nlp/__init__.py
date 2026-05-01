"""
NLP Module - Character Detection, Context Analysis, and Dialogue Parsing

Uses spaCy for named entity recognition and custom heuristics for
dialogue attribution, emotion detection, and voice trait inference.
"""

from .character_detector import CharacterDetector
from .context_analyzer import ContextAnalyzer
from .dialogue_parser import DialogueParser


def analyze_book(db, book):
    """
    Run full NLP analysis on a parsed book.

    1. Detect characters via NER and dialogue attribution
    2. Build dramatis personae with inferred voice traits
    3. Parse chapters into segments (dialogue/narration)
    4. Analyze context for each segment (emotion, emphasis, pacing)
    """
    from src.models import Chapter, Character, Segment, SegmentType

    detector = CharacterDetector()
    analyzer = ContextAnalyzer()
    parser = DialogueParser()

    # Collect all text for character detection
    chapters = db.query(Chapter).filter(Chapter.book_id == book.id).order_by(Chapter.number).all()
    full_text = "\n\n".join(ch.raw_text for ch in chapters)

    # Step 1: Detect characters
    characters_data = detector.detect_characters(full_text)
    character_map = {}  # name -> Character ORM object

    for char_data in characters_data:
        character = Character(
            book_id=book.id,
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
        db.flush()  # Get ID
        character_map[char_data["name"].lower()] = character
        for alias in char_data.get("aliases", []):
            character_map[alias.lower()] = character

    # Step 2: Parse each chapter into segments
    for chapter in chapters:
        segments_data = parser.parse_chapter(chapter.raw_text)

        for i, seg in enumerate(segments_data):
            # Find character for dialogue segments
            character_id = None
            if seg["type"] == "dialogue" and seg.get("speaker"):
                char = character_map.get(seg["speaker"].lower())
                if char:
                    character_id = char.id

            # Analyze context
            context = analyzer.analyze_segment(seg["text"], seg["type"])

            segment = Segment(
                chapter_id=chapter.id,
                character_id=character_id,
                text=seg["text"],
                segment_type=SegmentType(seg["type"]) if seg["type"] in SegmentType.__members__.values() else SegmentType.NARRATION,
                sequence_number=i,
                emotion=context.get("emotion"),
                emphasis=context.get("emphasis"),
                pacing=context.get("pacing"),
            )
            db.add(segment)

    book.total_characters = len(characters_data)
    db.flush()


__all__ = ["CharacterDetector", "ContextAnalyzer", "DialogueParser", "analyze_book"]
