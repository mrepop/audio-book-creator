"""
NLP Module - Character Detection, Context Analysis, and Dialogue Parsing

Pipeline:
  1. Detect characters via NER (character_detector)
  2. Split chapters into sentences (sentence_splitter)
  3. LLM-annotate each sentence with emotion, emphasis, speaker, vocal_direction
  4. Group annotated sentences into small segments (2-3 sentences max)
  5. Create Segment DB records with LLM-generated vocal directions
"""

import logging
from typing import List, Dict, Optional

from .character_detector import CharacterDetector
from .context_analyzer import ContextAnalyzer
from .dialogue_parser import DialogueParser
from .sentence_splitter import split_sentences, Sentence
from .llm_context_analyzer import LLMContextAnalyzer, SentenceAnnotation

logger = logging.getLogger(__name__)

# Max words per narration segment (prevents the old 3000-word blocks)
_MAX_NARRATION_SENTENCES = 3
_MAX_NARRATION_WORDS = 80


def analyze_book(db, book):
    """
    Run full NLP analysis on a parsed book.

    1. Detect characters via NER and dialogue attribution
    2. Build dramatis personae with inferred voice traits
    3. Split chapters into sentences
    4. LLM-annotate sentences with vocal directions
    5. Group into small segments and create DB records
    """
    from src.models import Chapter, Character, Segment, SegmentType, VoiceProfile
    from src.api.config import get_config

    config = get_config()
    nlp_cfg = config.nlp
    use_llm = getattr(nlp_cfg, 'llm_enabled', True)

    detector = CharacterDetector()
    fallback_analyzer = ContextAnalyzer()

    # Collect chapter data for character detection
    chapters = db.query(Chapter).filter(Chapter.book_id == book.id).order_by(Chapter.number).all()
    chapter_data = [{"number": ch.number, "text": ch.raw_text} for ch in chapters]

    # ---- Step 1: Detect characters (chapter-aware NER) ----
    logger.info(f"Detecting characters in book {book.id}...")
    characters_data = detector.detect_characters(chapter_data)
    character_map = {}  # name/alias/part -> Character ORM object
    character_names = [c["name"] for c in characters_data]

    seed_base = 42
    for i, char_data in enumerate(characters_data):
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
        for part in char_data["name"].lower().split():
            if len(part) >= 3 and part not in character_map:
                character_map[part] = character
        for alias in char_data.get("aliases", []):
            character_map[alias.lower()] = character
            for part in alias.lower().split():
                if len(part) >= 3 and part not in character_map:
                    character_map[part] = character

    logger.info(f"Created {len(characters_data)} characters with voice profiles")

    # ---- Step 2: Sentence-level analysis with LLM ----
    # Load spaCy model for sentence splitting (reuse from character detector)
    spacy_nlp = detector._get_nlp()

    llm_analyzer = None
    if use_llm:
        try:
            llm_model = getattr(nlp_cfg, 'llm_model', 'mlx-community/Qwen2.5-7B-Instruct-4bit')
            window_size = getattr(nlp_cfg, 'llm_context_window', 12)
            overlap = getattr(nlp_cfg, 'llm_context_overlap', 3)
            llm_analyzer = LLMContextAnalyzer(
                model_name=llm_model,
                window_size=window_size,
                overlap=overlap,
            )
            llm_analyzer.start()
            logger.info("LLM context analyzer started")
        except Exception as e:
            logger.warning(f"LLM context analyzer unavailable, using keyword fallback: {e}")
            llm_analyzer = None

    try:
        total_segments = 0
        for ch_idx, chapter in enumerate(chapters):
            logger.info(f"Analyzing chapter {ch_idx+1}/{len(chapters)}: {chapter.title or 'Untitled'}")

            # Split into sentences
            sentences = split_sentences(chapter.raw_text, spacy_nlp=spacy_nlp)

            if not sentences:
                logger.warning(f"  Chapter {chapter.number}: no sentences found")
                continue

            # LLM annotation (or fallback)
            if llm_analyzer:
                annotations = llm_analyzer.analyze_sentences(sentences, character_names)
            else:
                # Keyword fallback: analyze each sentence individually
                annotations = []
                for sent in sentences:
                    ctx = fallback_analyzer.analyze_segment(sent.text, "dialogue" if sent.is_dialogue else "narration")
                    annotations.append(SentenceAnnotation(
                        sentence_index=sent.index,
                        emotion=ctx.get("emotion", "neutral"),
                        emphasis=ctx.get("emphasis", "normal"),
                        pacing=ctx.get("pacing", "normal"),
                    ))

            # Group annotated sentences into segments
            grouped = _group_sentences_into_segments(sentences, annotations)

            for seq, group in enumerate(grouped):
                text = " ".join(s.text for s in group["sentences"])
                ann = group["annotation"]  # Primary annotation for the group

                # Resolve speaker to character
                character_id = None
                if ann.speaker and ann.speaker != "Narrator":
                    speaker_lower = ann.speaker.lower()
                    char = character_map.get(speaker_lower)
                    if not char:
                        for part in speaker_lower.split():
                            if len(part) >= 3 and part in character_map:
                                char = character_map[part]
                                break
                    if char:
                        character_id = char.id

                seg_type = SegmentType.DIALOGUE if group["is_dialogue"] else SegmentType.NARRATION

                segment = Segment(
                    chapter_id=chapter.id,
                    character_id=character_id,
                    detected_character_id=character_id,
                    text=text,
                    segment_type=seg_type,
                    sequence_number=seq,
                    emotion=ann.emotion,
                    emphasis=ann.emphasis,
                    pacing=ann.pacing,
                    vocal_direction=ann.vocal_direction or None,
                )
                db.add(segment)
                total_segments += 1

            logger.info(
                f"  Chapter {chapter.number}: {len(sentences)} sentences -> {len(grouped)} segments"
            )

    finally:
        if llm_analyzer:
            llm_analyzer.stop()

    book.total_characters = len(characters_data)
    db.flush()
    logger.info(f"Analysis complete: {len(characters_data)} characters, {total_segments} segments")


def _group_sentences_into_segments(
    sentences: List[Sentence],
    annotations: List[SentenceAnnotation],
) -> List[Dict]:
    """
    Group consecutive annotated sentences into small segments.

    Rules:
    - Dialogue sentences: each quoted block is its own segment
    - Narration: group up to 3 sentences with the same emotion/emphasis
    - Break on: speaker change, emotion change, emphasis change, paragraph boundary
    - Max ~80 words per narration segment
    """
    if not sentences:
        return []

    groups = []
    current_sents = []
    current_ann = None
    current_is_dialogue = None
    current_word_count = 0

    for sent, ann in zip(sentences, annotations):
        is_dlg = sent.is_dialogue
        word_count = len(sent.text.split())

        # Decide whether to break the current group
        should_break = False
        if current_ann is None:
            should_break = False  # First sentence
        elif is_dlg != current_is_dialogue:
            should_break = True  # Dialogue/narration boundary
        elif is_dlg:
            # Dialogue: break on speaker change
            if ann.speaker != current_ann.speaker:
                should_break = True
        else:
            # Narration: break on style change, paragraph boundary, or size limit
            if ann.emotion != current_ann.emotion:
                should_break = True
            elif ann.emphasis != current_ann.emphasis:
                should_break = True
            elif sent.is_paragraph_start and current_sents:
                should_break = True
            elif len(current_sents) >= _MAX_NARRATION_SENTENCES:
                should_break = True
            elif current_word_count + word_count > _MAX_NARRATION_WORDS:
                should_break = True

        if should_break and current_sents:
            groups.append({
                "sentences": current_sents,
                "annotation": current_ann,
                "is_dialogue": current_is_dialogue,
            })
            current_sents = []
            current_word_count = 0

        current_sents.append(sent)
        current_ann = ann
        current_is_dialogue = is_dlg
        current_word_count += word_count

    # Flush remaining
    if current_sents:
        groups.append({
            "sentences": current_sents,
            "annotation": current_ann,
            "is_dialogue": current_is_dialogue,
        })

    return groups


__all__ = [
    "CharacterDetector", "ContextAnalyzer", "DialogueParser",
    "LLMContextAnalyzer", "SentenceAnnotation",
    "analyze_book",
]
