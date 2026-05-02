"""Inspect actual text content for quote chars and structure."""
import sys, re, unicodedata
sys.path.insert(0, ".")
from src.api.database import SessionLocal
from src.models import Chapter

db = SessionLocal()
ch = db.query(Chapter).filter(Chapter.number == 5).first()
if not ch:
    print("No chapter 5")
    sys.exit(1)

text = ch.raw_text[:3000]

# Show unique non-ascii chars
print("=== Non-ASCII chars ===")
for c in sorted(set(text)):
    if ord(c) > 127:
        print(f"  U+{ord(c):04X} {unicodedata.name(c, '?')} -> {c!r}")

# Show first dialogue-like text
print("\n=== Dialogue samples ===")
count = 0
for m in re.finditer(r'[\u201c\u201d"\u2018\u2019\'].{10,120}[\u201c\u201d"\u2018\u2019\']', text):
    print(repr(m.group()[:120]))
    count += 1
    if count >= 5:
        break
if count == 0:
    print("  (none found)")

# Show raw text structure
print("\n=== Raw text (first 600 chars) ===")
print(repr(text[:600]))

db.close()
