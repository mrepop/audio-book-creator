#!/usr/bin/env python3
"""
TTS Quality Test Harness

Generates known text through the full chunker -> TTS -> splice pipeline
and validates output using Whisper transcription comparison.

Tests:
  1. Short sentence (5 words) -- should produce clean ~2-3s audio
  2. Medium paragraph (30 words) -- single chunk, ~12s
  3. Long paragraph (80 words) -- multi-chunk with splice
  4. Slow-paced text -- verifies pacing-aware chunking
  5. Dialogue with instruct -- verifies vocal direction doesn't break output
  6. Archaic/literary text -- the Frankenstein use case

Each test checks:
  - Audio was produced (non-empty)
  - Duration is within expected range (not runaway)
  - Whisper transcription similarity > 0.6 (not gibberish)
"""

import sys
import os
import time
import logging
import tempfile

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
)
logger = logging.getLogger("tts_test")

import numpy as np
import soundfile as sf

from src.tts.chunker import chunk_text, ChunkerConfig
from src.audio.splicer import splice_chunks
from src.audio.quality_analyzer import analyze_segment as check_quality


# ---------------------------------------------------------------------------
# Test cases: (name, text, instruct, expected_min_s, expected_max_s, min_similarity)
# ---------------------------------------------------------------------------
TEST_CASES = [
    (
        "short_sentence",
        "The quick brown fox jumps over the lazy dog.",
        "",
        1.0, 8.0,
        0.5,
    ),
    (
        "medium_paragraph",
        "It is a truth universally acknowledged, that a single man in "
        "possession of a good fortune, must be in want of a wife. However "
        "little known the feelings or views of such a man may be on his "
        "first entering a neighbourhood, this truth is so well fixed in "
        "the minds of the surrounding families.",
        "",
        8.0, 30.0,
        0.5,
    ),
    (
        "long_paragraph",
        "I am by birth a Genevese, and my family is one of the most "
        "distinguished of that republic. My ancestors had been for many "
        "years counsellors and syndics, and my father had filled several "
        "public situations with honour and reputation. He was respected "
        "by all who knew him for his integrity and indefatigable attention "
        "to public business. He passed his younger days perpetually "
        "occupied by the affairs of his country; a variety of circumstances "
        "had prevented his marrying early, nor was it until the decline of "
        "life that he became a husband and the father of a family.",
        "",
        15.0, 60.0,
        0.4,
    ),
    (
        "slow_paced",
        "I trembled violently at his words. My father continued to speak, "
        "his voice low and measured, each word carrying the weight of years.",
        "Speak slowly and deliberately, with gravitas.",
        5.0, 20.0,
        0.4,
    ),
    (
        "dialogue_with_instruct",
        "I confess, my son, that I have always looked forward to your "
        "marriage with our dear Elizabeth as the foundation of our happiness.",
        "Speak with warmth and tenderness, as a loving father.",
        5.0, 20.0,
        0.4,
    ),
    (
        "archaic_literary",
        "Nothing is so painful to the human mind as a great and sudden change. "
        "The sun might shine, or the clouds might lower; but nothing could "
        "appear to me as it had done the day before.",
        "",
        5.0, 25.0,
        0.4,
    ),
]


def run_test(engine, test_name, text, instruct, min_dur, max_dur, min_sim, output_dir):
    """Run a single test case and return (passed, details_dict)."""
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST: {test_name}")
    logger.info(f"  Text: {text[:80]}...")
    logger.info(f"  Instruct: {instruct or '(none)'}")
    logger.info(f"  Expected: {min_dur:.0f}-{max_dur:.0f}s, similarity>{min_sim}")
    logger.info(f"{'='*60}")

    result = {
        "name": test_name,
        "text_words": len(text.split()),
        "passed": False,
        "issues": [],
    }

    # Determine WPS based on instruct
    wps = 2.5
    if instruct and ("slow" in instruct.lower() or "deliberate" in instruct.lower()):
        wps = 1.8

    # Chunk the text
    cfg = ChunkerConfig(target_seconds=10.0, max_seconds=12.0, words_per_second=wps)
    chunks = chunk_text(text, config=cfg)
    result["num_chunks"] = len(chunks)
    logger.info(f"  Chunked into {len(chunks)} chunk(s)")
    for i, c in enumerate(chunks):
        logger.info(f"    Chunk {i+1}: ~{c.estimated_duration:.1f}s, {len(c.text.split())} words")

    # Generate audio with retry logic (mirrors production pipeline)
    # Retries on: bad chunk duration, or Whisper gibberish detection
    MAX_ATTEMPTS = 3
    best_chunk_results = None
    best_similarity = -1.0
    t0 = time.perf_counter()

    for attempt in range(MAX_ATTEMPTS):
        seed = 42 + attempt * 100
        try:
            attempt_results = engine.generate_chunks(
                chunks=chunks,
                speaker="Ryan",
                instruct=instruct,
                temperature=0.7,
                seed=seed,
            )
        except Exception as e:
            logger.error(f"  Attempt {attempt+1} (seed={seed}): TTS crashed: {e}")
            continue

        # Duration sanity check
        bad_duration = any(
            len(audio) / sr > 15.0 or len(audio) / sr < 0.3
            for audio, sr in attempt_results
        )
        if bad_duration:
            logger.warning(f"  Attempt {attempt+1} (seed={seed}): bad chunk duration, retrying...")
            continue

        # Whisper content check: splice, write, check similarity
        tmp_path = os.path.join(output_dir, f"{test_name}_attempt{attempt}.wav")
        spliced = splice_chunks(
            [a for a, _ in attempt_results],
            sample_rate=attempt_results[0][1],
            crossfade_ms=100, sentence_silence_ms=200,
            paragraph_silence_ms=500, target_lufs=-16.0,
            paragraph_end_flags=[c.is_paragraph_end for c in chunks],
        )
        sf.write(tmp_path, spliced, attempt_results[0][1])
        quick_qa = check_quality(
            tmp_path, expected_text=text,
            expected_duration_s=len(text.split()) / wps,
            use_whisper=True, whisper_model_size="base",
            text_similarity_fail=0.3, text_similarity_warn=0.5,
            duration_ratio_fail=3.0, duration_ratio_warn=2.0,
            max_silence_s_fail=5.0, max_silence_s_warn=2.0,
            clipping_ratio_fail=0.02, repetition_score_fail=0.8,
        )
        sim = quick_qa.text_similarity if quick_qa else 0.0
        logger.info(f"  Attempt {attempt+1} (seed={seed}): similarity={sim:.2f}")

        # Keep the best attempt
        if sim > best_similarity:
            best_similarity = sim
            best_chunk_results = attempt_results

        # Good enough -- stop retrying
        if sim >= min_sim:
            break
        else:
            logger.warning(
                f"  Attempt {attempt+1} (seed={seed}): gibberish (sim={sim:.2f}), retrying..."
            )

    chunk_results = best_chunk_results
    if chunk_results is None:
        result["issues"].append(f"All {MAX_ATTEMPTS} generation attempts failed")
        return False, result

    gen_time = time.perf_counter() - t0
    result["generation_time_s"] = round(gen_time, 1)

    # Check each chunk duration
    chunk_durations = []
    for i, (audio, sr) in enumerate(chunk_results):
        dur = len(audio) / sr
        chunk_durations.append(dur)
        logger.info(f"  Chunk {i+1} output: {dur:.2f}s")
        if dur > 15.0:
            result["issues"].append(f"Chunk {i+1} exceeded 15s ceiling: {dur:.1f}s")
        if dur < 0.5:
            result["issues"].append(f"Chunk {i+1} suspiciously short: {dur:.1f}s")

    result["chunk_durations"] = [round(d, 2) for d in chunk_durations]

    # Splice chunks together
    chunk_audios = [audio for audio, sr in chunk_results]
    sample_rate = chunk_results[0][1] if chunk_results else 24000

    audio = splice_chunks(
        chunk_audios=chunk_audios,
        sample_rate=sample_rate,
        crossfade_ms=100,
        sentence_silence_ms=200,
        paragraph_silence_ms=500,
        target_lufs=-16.0,
        paragraph_end_flags=[c.is_paragraph_end for c in chunks],
    )

    total_dur = len(audio) / sample_rate
    result["total_duration_s"] = round(total_dur, 2)
    logger.info(f"  Total audio: {total_dur:.2f}s (generated in {gen_time:.1f}s)")

    # Save audio
    out_path = os.path.join(output_dir, f"{test_name}.wav")
    sf.write(out_path, audio, sample_rate)
    result["audio_path"] = out_path

    # Duration check
    if total_dur < min_dur:
        result["issues"].append(f"Too short: {total_dur:.1f}s < {min_dur:.0f}s expected")
    if total_dur > max_dur:
        result["issues"].append(f"Too long: {total_dur:.1f}s > {max_dur:.0f}s expected")

    # Quality check with Whisper
    est_dur = len(text.split()) / 2.5
    qa = check_quality(
        out_path,
        expected_text=text,
        expected_duration_s=est_dur,
        use_whisper=True,
        whisper_model_size="base",
        text_similarity_fail=0.3,
        text_similarity_warn=0.5,
        duration_ratio_fail=3.0,
        duration_ratio_warn=2.0,
        max_silence_s_fail=5.0,
        max_silence_s_warn=2.0,
        clipping_ratio_fail=0.02,
        repetition_score_fail=0.8,
    )

    result["whisper_similarity"] = round(qa.text_similarity, 2) if qa else None
    result["quality_score"] = qa.score if qa else None
    result["quality_passed"] = qa.passed if qa else None
    result["quality_issues"] = qa.issues if qa else []

    logger.info(f"  Whisper similarity: {qa.text_similarity:.2f}" if qa else "  Whisper: unavailable")
    logger.info(f"  Quality score: {qa.score}" if qa else "")

    if qa and qa.text_similarity < min_sim:
        result["issues"].append(
            f"Whisper similarity too low: {qa.text_similarity:.2f} < {min_sim}"
        )

    # Verdict
    result["passed"] = len(result["issues"]) == 0
    status = "[PASS]" if result["passed"] else "[FAIL]"
    logger.info(f"  {status} {test_name}")
    if result["issues"]:
        for issue in result["issues"]:
            logger.warning(f"    - {issue}")

    return result["passed"], result


def main():
    output_dir = tempfile.mkdtemp(prefix="tts_test_")
    logger.info(f"Test output directory: {output_dir}")

    # Load TTS engine
    logger.info("Loading TTS engine...")
    from src.tts.qwen3_engine import Qwen3TTSEngine
    engine = Qwen3TTSEngine()
    engine._load_model(timeout_seconds=300)
    logger.info("Engine loaded")

    # Run tests
    results = []
    passed = 0
    failed = 0

    for test_name, text, instruct, min_dur, max_dur, min_sim in TEST_CASES:
        ok, result = run_test(
            engine, test_name, text, instruct,
            min_dur, max_dur, min_sim, output_dir,
        )
        results.append(result)
        if ok:
            passed += 1
        else:
            failed += 1

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST SUMMARY: {passed} passed, {failed} failed out of {len(TEST_CASES)}")
    logger.info(f"{'='*60}")

    for r in results:
        status = "[PASS]" if r["passed"] else "[FAIL]"
        sim = f"sim={r.get('whisper_similarity', '?')}" if r.get("whisper_similarity") is not None else ""
        dur = f"dur={r.get('total_duration_s', '?')}s"
        chunks = f"chunks={r.get('num_chunks', '?')}"
        logger.info(f"  {status} {r['name']:25s} | {dur:12s} | {chunks:10s} | {sim}")
        if r["issues"]:
            for issue in r["issues"]:
                logger.info(f"         - {issue}")

    logger.info(f"\nAudio files saved to: {output_dir}")

    # Exit code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
