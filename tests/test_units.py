"""Unit tests — no gated data, no Ollama, no network. Run: pytest -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reward import score_answer, extract_final_answer
from src import chunking
from src.common import (parse_year_month, parse_source_files, question_years,
                        question_quarter)
from src.matching import answer_in_text, numeric_claims, answer_numbers
from src.retrieve import build_where

# A tiny Markdown fragment shaped like a Treasury Bulletin page.
SAMPLE = """INTERNATIONAL STATISTICS

Some introductory prose about foreign currency positions that runs on for a
little while so it forms its own paragraph block in the chunker.

| Country | Amount |
| --- | --- |
| Japan | 935851121560 |
| Switzerland | 12345 |

EXCHANGE STABILIZATION FUND

| Item | Value |
| --- | --- |
| Total assets | 4.815 |
"""


# ---- metadata / filename parsing -------------------------------------------
def test_parse_year_month():
    assert parse_year_month("treasury_bulletin_2023_03.txt") == ("2023", "03")
    assert parse_year_month("garbage.txt") == (None, None)


def test_parse_source_files_handles_messy_cells():
    assert parse_source_files("treasury_bulletin_2016_09.txt") == ["treasury_bulletin_2016_09.txt"]
    multi = "['treasury_bulletin_2016_12.txt', 'treasury_bulletin_2017_06.txt']"
    assert parse_source_files(multi) == ["treasury_bulletin_2016_12.txt", "treasury_bulletin_2017_06.txt"]
    crlf = "treasury_bulletin_1941_01.txt\r\ntreasury_bulletin_1954_02.txt"
    assert len(parse_source_files(crlf)) == 2


def test_question_year_and_quarter_parsing():
    assert question_years("value as of March 31, 2023") == ["2023"]
    assert question_years("nothing here") == []
    assert question_quarter("as of March 31") == "03"
    assert question_quarter("in Q3 of that year") == "09"


# ---- metadata filter (the engineered retriever's core trick) ---------------
def test_build_where_expands_for_fiscal_lag():
    where = build_where("total outstanding debt at year-end for CY 2022")
    assert where == {"year": {"$in": ["2022", "2023"]}}  # +1 for publication lag


def test_build_where_none_when_no_year_or_month():
    assert build_where("what is the total debt") is None


# ---- chunking: baseline slices tables, engineered keeps them ---------------
def test_baseline_chunks_are_fixed_width():
    chunks = chunking.chunk_baseline(SAMPLE)
    assert all(len(c) <= 1000 for c in chunks)


def test_engineered_keeps_table_rows_together_and_attaches_section():
    chunks = chunking.chunk_engineered(SAMPLE)
    japan = [c for c in chunks if "Japan | 935851121560" in c]
    assert japan, "the Japan table row must survive intact in some chunk"
    # a full table row (header + data) should co-occur, not be split mid-row
    assert "| Country | Amount |" in japan[0]
    # ALL-CAPS section heading is detected and prepended somewhere
    assert any("INTERNATIONAL STATISTICS" in c for c in chunks)


# ---- official reward.py scorer ---------------------------------------------
def test_reward_scorer_tag_and_tolerance():
    assert score_answer("935851121560", "<FINAL_ANSWER>935851121560</FINAL_ANSWER>", 0.01) == 1.0
    # within 1% passes, outside fails
    assert score_answer("1000", "<FINAL_ANSWER>1005</FINAL_ANSWER>", 0.01) == 1.0
    assert score_answer("1000", "<FINAL_ANSWER>1200</FINAL_ANSWER>", 0.01) == 0.0
    assert extract_final_answer("blah <FINAL_ANSWER>42</FINAL_ANSWER>") == "42"


# ---- number-containment matching (cell relevance + groundedness) -----------
def test_answer_in_text_and_claims():
    assert answer_in_text("935851121560", SAMPLE)
    assert answer_in_text("1,169.41 million", "the table shows 1169.41 here")
    assert not answer_in_text("999999", SAMPLE)
    # bare years are not counted as numeric claims; real figures are
    assert numeric_claims("reported in 2023 the value was 4.815") == ["4.815"]
    assert answer_numbers("$1,234.5 million") == [1234.5]
