#!/usr/bin/env python3
"""
Check audio quality of generated segments.

Analyzes WAV files in storage/temp/ for Qwen3 artifacts:
  - Runaway generation (too long)
  - Gibberish (Whisper transcription mismatch)
  - Stuttering/repetition
  - Silence gaps
  - Clipping

Usage:
    # Analyze all segment WAVs from the latest generation
    venv/bin/python scripts/check_audio_quality.py

    # Analyze a specific directory
    venv/bin/python scripts/check_audio_quality.py --dir storage/temp

    # Skip Whisper (faster, signal-only checks)
    venv/bin/python scripts/check_audio_quality.py --no-whisper

    # Analyze with text from the database (matches segment audio to expected text)
    venv/bin/python scripts/check_audio_quality.py --from-db
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

def main():
    parser = argparse.ArgumentParser(description="Analyze TTS audio quality")
    parser.add_argument("--dir", default="storage/temp", help="Directory with WAV files")
    parser.add_argument("--no-whisper", action="store_true", help="Skip Whisper transcription")
    parser.add_argument("--from-db", action="store_true", help="Load expected text from database")
    parser.add_argument("--whisper-model", default="base", help="Whisper model size")
    parser.add_argument("--limit", type=int, default=0, help="Max files to analyze (0=all)")
    args = parser.parse_args()

    from src.audio.quality_analyzer import analyze_segment, QualityReport

    wav_dir = Path(args.dir)
    if not wav_dir.exists():
        print(f"Directory not found: {wav_dir}")
        return

    wav_files = sorted(wav_dir.glob("*seg*.wav"))
    if not wav_files:
        wav_files = sorted(wav_dir.glob("*.wav"))

    if not wav_files:
        print(f"No WAV files found in {wav_dir}")
        return

    if args.limit > 0:
        wav_files = wav_files[:args.limit]

    # Load expected texts from DB if requested
    expected_texts = {}
    if args.from_db:
        expected_texts = _load_texts_from_db()

    print(f"Analyzing {len(wav_files)} files (whisper={'OFF' if args.no_whisper else args.whisper_model})...")
    print()

    reports: list[QualityReport] = []
    for wav_path in wav_files:
        # Try to extract expected text from filename -> DB mapping
        expected = expected_texts.get(wav_path.name, "")

        report = analyze_segment(
            str(wav_path),
            expected_text=expected,
            use_whisper=not args.no_whisper and bool(expected),
            whisper_model_size=args.whisper_model,
        )
        reports.append(report)

    # Summary
    print()
    print("=" * 70)
    print("QUALITY SUMMARY")
    print("=" * 70)

    passed = [r for r in reports if r.passed]
    failed = [r for r in reports if not r.passed]
    warned = [r for r in reports if r.warnings and r.passed]

    print(f"  Total files: {len(reports)}")
    print(f"  [PASS]: {len(passed)}")
    print(f"  [FAIL]: {len(failed)}")
    print(f"  [WARN]: {len(warned)} (passed with warnings)")
    print(f"  Avg score: {sum(r.score for r in reports) / max(len(reports), 1):.0f}/100")
    print()

    if failed:
        print("FAILED FILES:")
        for r in failed:
            print(f"  {Path(r.path).name}: score={r.score:.0f}")
            for issue in r.issues:
                print(f"    [ISSUE] {issue}")
        print()

    # Issue type breakdown
    all_issues = []
    for r in reports:
        all_issues.extend(r.issues)
        all_issues.extend(r.warnings)

    if all_issues:
        print("ISSUE BREAKDOWN:")
        from collections import Counter
        types = Counter()
        for issue in all_issues:
            issue_type = issue.split(":")[0]
            types[issue_type] += 1
        for t, count in types.most_common():
            print(f"  {t}: {count}")


def _load_texts_from_db() -> dict[str, str]:
    """Load segment texts from the database, keyed by expected WAV filename."""
    try:
        from src.api.database import get_db_context
        from src.models import Segment, Chapter

        texts = {}
        with get_db_context() as db:
            segments = db.query(Segment).filter(Segment.audio_path.isnot(None)).all()
            for seg in segments:
                if seg.audio_path:
                    filename = Path(seg.audio_path).name
                    texts[filename] = seg.user_text_override or seg.text
        print(f"Loaded {len(texts)} segment texts from database")
        return texts
    except Exception as e:
        print(f"Could not load texts from DB: {e}")
        return {}


if __name__ == "__main__":
    main()
