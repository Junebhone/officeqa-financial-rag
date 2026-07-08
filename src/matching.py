"""Number-containment matching, built on the official reward.py extractor.

Used for two things the LLM judge previously did, but now deterministically:
  - CELL-level retrieval relevance: does a chunk literally contain the gold value?
  - Groundedness: is each number the model asserted actually present in context?

Reusing reward.py's number extraction means "$1,234.5 million" in an answer and
"1,234.5" in a table normalize identically to the official scorer.
"""
import re

from reward import extract_numbers_with_context

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*%?")


def answer_numbers(text: str) -> list[float]:
    try:
        return [n for n, *_ in extract_numbers_with_context(text)]
    except Exception:
        out = []
        for m in NUM_RE.findall(text):
            try:
                out.append(float(m.rstrip("%").replace(",", "")))
            except ValueError:
                pass
        return out


def answer_in_text(answer: str, text: str) -> bool:
    """True if every numeric value in `answer` appears in `text` (after reward.py
    normalization); for non-numeric answers, case-insensitive substring."""
    answer = str(answer).strip()
    nums = answer_numbers(answer)
    if not nums:
        return answer.lower() in text.lower()
    try:
        text_nums = {n for n, *_ in extract_numbers_with_context(text)}
    except Exception:
        return False
    return all(any(abs(a - t) < 1e-9 for t in text_nums) for a in nums)


def numeric_claims(text: str) -> list[str]:
    """Numbers the model asserts (skip bare 4-digit years, which aren't claims)."""
    claims = []
    for m in NUM_RE.findall(text):
        raw = m.rstrip("%").replace(",", "")
        try:
            v = float(raw)
        except ValueError:
            continue
        if 1930 <= v <= 2030 and "." not in raw:
            continue
        claims.append(m)
    return claims
