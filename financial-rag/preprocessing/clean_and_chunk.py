"""Clean HTML -> text -> sections -> chunks.

10-Ks follow a standard structure with Items 1, 1A, 2, ..., 15. We use those
headers as section boundaries, which is much better than blind fixed-size chunks.

Each output chunk carries metadata: section, source_type, company, chunk_id.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

from bs4 import BeautifulSoup
import tiktoken

from configs.config import (
    TICKER, DATA_RAW, CLEAN_TEXT_PATH, SECTIONS_PATH, CHUNKS_PATH,
    CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS, FORM_TYPE,
)


# Standard 10-K item headers. Order matters — used to split the doc.
# We match loosely (case-insensitive, tolerant of whitespace/periods) because
# filers are inconsistent.
ITEM_HEADERS = [
    ("item_1",   "Business"),
    ("item_1a",  "Risk Factors"),
    ("item_1b",  "Unresolved Staff Comments"),
    ("item_2",   "Properties"),
    ("item_3",   "Legal Proceedings"),
    ("item_4",   "Mine Safety Disclosures"),
    ("item_5",   "Market for Registrant's Common Equity"),
    ("item_6",   "Selected Financial Data"),
    ("item_7",   "Management's Discussion and Analysis"),
    ("item_7a",  "Quantitative and Qualitative Disclosures About Market Risk"),
    ("item_8",   "Financial Statements and Supplementary Data"),
    ("item_9",   "Changes in and Disagreements with Accountants"),
    ("item_9a",  "Controls and Procedures"),
    ("item_9b",  "Other Information"),
    ("item_10",  "Directors, Executive Officers and Corporate Governance"),
    ("item_11",  "Executive Compensation"),
    ("item_12",  "Security Ownership"),
    ("item_13",  "Certain Relationships and Related Transactions"),
    ("item_14",  "Principal Accountant Fees and Services"),
    ("item_15",  "Exhibits and Financial Statement Schedules"),
]


@dataclass
class Chunk:
    chunk_id: str
    text: str
    company: str
    source_type: str
    section_id: str
    section_name: str


def html_to_text(html_path: Path) -> str:
    """Strip HTML/XBRL tags, keep readable text."""
    raw = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "lxml")

    # Remove script/style and XBRL metadata that some filings embed
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # Collapse excessive blank lines and non-breaking spaces
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_sections(text: str) -> List[dict]:
    """Split the cleaned 10-K text on Item headers.

    Returns list of {section_id, section_name, text}. Text before the first
    recognized Item goes into a 'preamble' section (cover page, TOC, etc.).
    """
    # Build a regex that matches any item header. Tolerates "ITEM 1A.", "Item 1A",
    # "ITEM\xa01A", etc. We look for these on their own line or with surrounding ws.
    patterns = []
    for sid, _ in ITEM_HEADERS:
        num = sid.replace("item_", "")
        # match "Item 1A" with flexible spacing and optional period
        patterns.append((sid, re.compile(
            rf"(?im)^\s*item\s*{re.escape(num)}\.?\s*[:\-\.]?",
        )))

    # Find all match positions: (position, section_id)
    matches = []
    for sid, pat in patterns:
        for m in pat.finditer(text):
            matches.append((m.start(), sid))
    matches.sort()

    # Dedupe: for each section_id, keep the LAST occurrence (TOC appears first,
    # actual content appears later in the document).
    last_pos_per_section = {}
    for pos, sid in matches:
        last_pos_per_section[sid] = pos
    # But we need them in document order now
    ordered = sorted(last_pos_per_section.items(), key=lambda x: x[1])

    # Build sections from span between consecutive match positions
    sections = []
    # anything before first real section start
    if ordered and ordered[0][1] > 0:
        preamble = text[:ordered[0][1]].strip()
        if preamble:
            sections.append({
                "section_id": "preamble",
                "section_name": "Cover / TOC",
                "text": preamble,
            })

    name_map = dict(ITEM_HEADERS)
    for i, (sid, start) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else len(text)
        body = text[start:end].strip()
        if len(body) < 50:  # skip near-empty sections
            continue
        sections.append({
            "section_id": sid,
            "section_name": name_map.get(sid, sid),
            "text": body,
        })

    return sections


def chunk_sections(sections: List[dict], company: str = TICKER) -> List[Chunk]:
    """Token-aware chunking within each section, with overlap."""
    enc = tiktoken.get_encoding("cl100k_base")
    chunks: List[Chunk] = []

    for sec in sections:
        tokens = enc.encode(sec["text"])
        if not tokens:
            continue

        step = CHUNK_TOKENS - CHUNK_OVERLAP_TOKENS
        idx = 0
        sub = 0
        while idx < len(tokens):
            window = tokens[idx: idx + CHUNK_TOKENS]
            chunk_text = enc.decode(window)
            chunks.append(Chunk(
                chunk_id=f"{company}-{sec['section_id']}-{sub:03d}",
                text=chunk_text,
                company=company,
                source_type=FORM_TYPE,
                section_id=sec["section_id"],
                section_name=sec["section_name"],
            ))
            idx += step
            sub += 1

    return chunks


def main():
    # Find the downloaded filing
    filings_dir = DATA_RAW / "sec-edgar-filings" / TICKER / FORM_TYPE
    accession_dirs = sorted([d for d in filings_dir.iterdir() if d.is_dir()])
    latest = accession_dirs[-1]
    candidates = list(latest.glob("*.htm*")) or list(latest.glob("*.txt"))
    primary = max(candidates, key=lambda p: p.stat().st_size)
    print(f"Processing: {primary}")

    # Clean
    text = html_to_text(primary)
    CLEAN_TEXT_PATH.write_text(text, encoding="utf-8")
    print(f"Clean text: {len(text):,} chars -> {CLEAN_TEXT_PATH}")

    # Sections
    sections = split_into_sections(text)
    SECTIONS_PATH.write_text(json.dumps(sections, indent=2), encoding="utf-8")
    print(f"Sections: {len(sections)} -> {SECTIONS_PATH}")
    for s in sections:
        print(f"  {s['section_id']:10s} {s['section_name']:50s} {len(s['text']):>8,} chars")

    # Chunks
    chunks = chunk_sections(sections)
    CHUNKS_PATH.write_text(
        json.dumps([asdict(c) for c in chunks], indent=2),
        encoding="utf-8",
    )
    print(f"Chunks: {len(chunks)} -> {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
