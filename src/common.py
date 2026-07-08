"""Shared helpers: filename/metadata parsing, answer-key loading, Ollama calls."""
import ast
import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests

import config

# --- Treasury Bulletin filenames: treasury_bulletin_YYYY_MM.txt --------------
FNAME_RE = re.compile(r"treasury_bulletin_(\d{4})_(\d{2})")
MONTHS = {
    "january": "03", "february": "03", "march": "03",  # bulletins are quarterly;
    "april": "06", "may": "06", "june": "06",           # map a month name to the
    "july": "09", "august": "09", "september": "09",     # quarter-end file it lands in
    "october": "12", "november": "12", "december": "12",
}


def parse_year_month(filename: str):
    """'treasury_bulletin_2023_03.txt' -> ('2023', '03')."""
    m = FNAME_RE.search(str(filename))
    return (m.group(1), m.group(2)) if m else (None, None)


def parse_source_files(value) -> list[str]:
    """Answer-key `source_files` cells are messy: python-list strings, or
    newline/CRLF/comma separated. Return a clean list of bare .txt filenames."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    s = str(value).strip()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            s = "\n".join(str(x) for x in parsed)
    except (ValueError, SyntaxError):
        pass
    parts = re.split(r"[,\r\n]+", s)
    return [p.strip() for p in parts if p.strip().endswith(".txt")]


def question_years(question: str) -> list[str]:
    """Extract 4-digit years (2015-2025) mentioned in a question -> used by the
    engineered retriever to build a Year metadata filter."""
    yrs = set(re.findall(r"\b(20\d{2})\b", question))
    return sorted(y for y in yrs if y in config.YEARS)


def question_quarter(question: str):
    """Map any month name in the question to its quarterly bulletin ('03'..'12')."""
    q = question.lower()
    for name, quarter in MONTHS.items():
        if name in q:
            return quarter
    m = re.search(r"\bq([1-4])\b", q)
    if m:
        return {"1": "03", "2": "06", "3": "09", "4": "12"}[m.group(1)]
    return None


# --- Answer key --------------------------------------------------------------
def load_eval_set() -> pd.DataFrame:
    """Load officeqa_full.csv and keep only questions whose ENTIRE set of source
    documents lives inside the corpus we actually indexed (config.YEARS).

    A question is excluded if it needs a document we didn't download, otherwise
    it would be an automatic retrieval miss that unfairly punishes the system."""
    df = pd.read_csv(config.ANSWER_KEY_CSV)
    have = {p.name for p in config.CORPUS_DIR.glob("*.txt")}
    rows = []
    for _, r in df.iterrows():
        gold = parse_source_files(r["source_files"])
        if gold and all(f in have for f in gold):
            rows.append({
                "uid": r["uid"],
                "question": r["question"],
                "answer": str(r["answer"]),
                "gold_files": gold,
                "difficulty": r.get("difficulty", ""),
            })
    return pd.DataFrame(rows)


# --- Ollama generation -------------------------------------------------------
def ollama_generate(prompt: str, system: str = "", json_mode: bool = False) -> str:
    payload = {
        "model": config.GEN_MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": config.GEN_TEMPERATURE, "num_ctx": config.GEN_NUM_CTX},
    }
    if json_mode:
        payload["format"] = "json"
    r = requests.post(f"{config.OLLAMA_URL}/api/generate", json=payload, timeout=600)
    r.raise_for_status()
    return r.json().get("response", "").strip()


def extract_json(text: str) -> dict:
    """Best-effort JSON parse from an LLM response."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


# --- Numeric comparison for factual accuracy ---------------------------------
def extract_number(text: str):
    """Pull the most salient numeric value from a string, ignoring $, commas, %."""
    if text is None:
        return None
    cleaned = str(text).replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if not nums:
        return None
    # Prefer the longest numeric token (usually the actual figure, not a year/rank).
    return float(max(nums, key=len))


def numeric_match(gold: str, pred: str, tol: float = config.FACTUAL_REL_TOLERANCE) -> bool | None:
    g = extract_number(gold)
    p = extract_number(pred)
    if g is None or p is None:
        return None  # not a numeric comparison -> caller falls back to LLM judge
    if g == 0:
        return abs(p) <= tol
    return abs(p - g) / abs(g) <= tol
