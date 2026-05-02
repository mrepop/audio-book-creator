"""Inspect DB state for debugging."""
import sys
sys.path.insert(0, ".")
from src.api.database import SessionLocal
from src.models import *

db = SessionLocal()

books = db.query(Book).all()
for b in books:
    print(f"Book: {b.title} | parsed={b.is_parsed} analyzed={b.is_analyzed} chapters={b.total_chapters} chars={b.total_characters}")

chars = db.query(Character).all()
print(f"\nCharacters ({len(chars)}):")
for c in chars[:25]:
    print(f"  name={c.name!r} role={c.role} gender={c.inferred_gender} dialogue={c.dialogue_count}")

seg_count = db.query(Segment).count()
segs = db.query(Segment).limit(10).all()
print(f"\nSegments ({seg_count} total), first 10:")
for s in segs:
    stype = s.segment_type.value if s.segment_type else "?"
    txt = s.text[:80].replace("\n", " ")
    print(f"  [{stype}] char_id={s.character_id} emotion={s.emotion} | {txt}")

db.close()
