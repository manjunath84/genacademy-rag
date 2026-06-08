"""Spike: parse-quality gate on the 19.3 MB guidebook (production corpus, not eval-blocking).

Checks design.md §9: char density not mostly whitespace/garbled, expected headings
present, mean chunk length above a floor. PASS => pypdf is enough; FAIL => OCR fallback.
"""
import statistics
import time
from pathlib import Path

from pypdf import PdfReader

PDF = Path(
    "/Users/manjunathans/projects/GenAcademy/CuratedRAGMaterials/"
    "Mastering-Agentic-AI-Getting-Started-Guidebook.pdf"
)

t0 = time.time()
reader = PdfReader(str(PDF))
n_pages = len(reader.pages)
texts = [(p.extract_text() or "") for p in reader.pages]
elapsed = time.time() - t0

total_chars = sum(len(t) for t in texts)
nonspace = sum(len(t.replace(" ", "").replace("\n", "")) for t in texts)
per_page = [len(t) for t in texts]
empty_pages = sum(1 for n in per_page if n < 50)
mean_chars = statistics.mean(per_page) if per_page else 0

# crude "garbled" signal: ratio of alphanumeric+punct to total non-space chars
allchars = "".join(texts)
printable = sum(1 for c in allchars if c.isalnum() or c in " .,;:!?-()[]'\"\n/")
printable_ratio = printable / max(1, len(allchars))

# heading signal: lines that look like section headers
import re
lines = [ln.strip() for ln in allchars.splitlines() if ln.strip()]
heading_like = sum(1 for ln in lines if re.match(r"^(chapter|section|\d+\.|\d+\s)|^[A-Z][A-Za-z ]{3,40}$", ln, re.I))

print(f"file              : {PDF.name}")
print(f"size              : {PDF.stat().st_size/1e6:.1f} MB")
print(f"pages             : {n_pages}")
print(f"extract time      : {elapsed:.1f} s")
print(f"total chars       : {total_chars:,}")
print(f"non-space chars   : {nonspace:,}")
print(f"mean chars/page   : {mean_chars:.0f}")
print(f"empty pages (<50) : {empty_pages}  ({100*empty_pages/max(1,n_pages):.0f}%)")
print(f"printable ratio   : {printable_ratio:.3f}")
print(f"heading-like lines: {heading_like}")

# Gate verdict (design.md §9 thresholds, conservative)
fail = []
if mean_chars < 200:
    fail.append("mean chars/page < 200 (likely image-heavy/scanned)")
if printable_ratio < 0.90:
    fail.append("printable ratio < 0.90 (garbled extraction)")
if empty_pages / max(1, n_pages) > 0.30:
    fail.append(">30% empty pages")
if heading_like < 5:
    fail.append("almost no heading structure detected")

print()
if fail:
    print("VERDICT: FAIL -> OCR fallback (pdf2image+pytesseract / marker) or exclude-and-log")
    for f in fail:
        print(f"  - {f}")
else:
    print("VERDICT: PASS -> pypdf extraction is adequate for the production corpus")
