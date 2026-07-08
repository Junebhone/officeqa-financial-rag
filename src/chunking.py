"""Two chunking strategies.

BASELINE  - blind fixed-width character windows. Cheap, but happily slices a
            financial table in half so no chunk holds a full row+header.

ENGINEERED - structure-aware. Splits the Markdown into blocks (paragraphs and
             *whole* pipe-table row groups), packs them up to a size budget with
             overlap, and prepends the nearest ALL-CAPS section header so an
             isolated table still carries its context ("EXCHANGE STABILIZATION
             FUND ... | ESF-1 Balance Sheet | ...").
"""
import re

import config

SECTION_RE = re.compile(r"^[A-Z][A-Z0-9 ,.&'/()-]{6,}$")  # ALL-CAPS heading line
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")               # markdown pipe row


def chunk_baseline(text: str) -> list[str]:
    size, overlap = config.BASELINE_CHUNK_CHARS, config.BASELINE_CHUNK_OVERLAP
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size].strip()]


def _blocks(text: str):
    """Yield structural blocks: consecutive table rows stay glued together;
    blank-line-separated prose becomes its own block."""
    lines = text.splitlines()
    buf, buf_is_table = [], False
    for line in lines:
        is_table = bool(TABLE_ROW_RE.match(line))
        if line.strip() == "" and not is_table:
            if buf:
                yield "\n".join(buf); buf, buf_is_table = [], False
            continue
        if buf and is_table != buf_is_table:
            yield "\n".join(buf); buf, buf_is_table = [], is_table
        buf.append(line); buf_is_table = is_table
    if buf:
        yield "\n".join(buf)


def chunk_engineered(text: str) -> list[str]:
    size, overlap = config.ENGINEERED_CHUNK_CHARS, config.ENGINEERED_CHUNK_OVERLAP
    chunks, cur, cur_len, section = [], [], 0, ""

    def flush():
        nonlocal cur, cur_len
        if not cur:
            return
        body = "\n".join(cur)
        header = f"[Section: {section}]\n" if section else ""
        chunks.append(header + body)
        # carry a tail of the current chunk forward as overlap for continuity
        if overlap > 0:
            tail = body[-overlap:]
            cur, cur_len = [tail], len(tail)
        else:
            cur, cur_len = [], 0

    for block in _blocks(text):
        stripped = block.strip()
        # Track the running section header (short ALL-CAPS non-table lines).
        if not TABLE_ROW_RE.match(block) and len(stripped) < 80 and SECTION_RE.match(stripped):
            section = stripped
        if cur_len + len(block) > size and cur_len > 0:
            flush()
        # A single oversized table block gets split on row boundaries, not chars.
        if len(block) > size:
            rows = block.splitlines()
            row_buf, row_len = [], 0
            for row in rows:
                if row_len + len(row) > size and row_buf:
                    chunks.append((f"[Section: {section}]\n" if section else "") + "\n".join(row_buf))
                    row_buf, row_len = [], 0
                row_buf.append(row); row_len += len(row) + 1
            if row_buf:
                cur.append("\n".join(row_buf)); cur_len += row_len
        else:
            cur.append(block); cur_len += len(block)
    flush()
    return [c for c in chunks if c.strip()]
