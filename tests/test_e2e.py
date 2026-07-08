"""End-to-end pipeline test — synthetic corpus, mocked LLM, no gated data.

Exercises the real path: chunk -> embed -> ChromaDB (with year/month metadata)
-> engineered retrieval (metadata filter) -> generation (mocked) -> the 6-metric
evaluator. Uses the actual embedding models (cached locally) but no Ollama and
no network, so it runs in CI.

Run: pytest -q tests/test_e2e.py
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src import build_index, generate, retrieve
from src.evaluate import evaluate_system
from src.retrieve import retrieve_engineered

# Gold doc for the 2023 question.
DOC_2023 = """INTERNATIONAL STATISTICS

Foreign currency positions reported by the U.S. Treasury for the quarter.

| Currency | Amount |
| --- | --- |
| Japanese Yen | 935851121560 |
| Swiss Franc | 12345678 |
"""

# Textually near-identical to the gold doc but from an OUT-OF-FILTER year — the
# metadata filter must exclude it despite high embedding similarity.
DOC_2020 = """INTERNATIONAL STATISTICS

Foreign currency positions reported by the U.S. Treasury for the quarter.

| Currency | Amount |
| --- | --- |
| Japanese Yen | 777777777777 |
| Swiss Franc | 88888888 |
"""

# In-filter (via year+1 lag) but a different topic — a weak distractor that
# should rank below the gold doc.
DOC_2024 = """SPECIAL REPORTS

Trust fund activity for airports, black lung, and vaccine injury compensation.

| Trust Fund | Balance |
| --- | --- |
| Airport and Airway | 33333333 |
| Black Lung | 44444444 |
"""


@pytest.fixture
def tiny_index(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "treasury_bulletin_2020_03.txt").write_text(DOC_2020)
    (corpus / "treasury_bulletin_2023_03.txt").write_text(DOC_2023)
    (corpus / "treasury_bulletin_2024_03.txt").write_text(DOC_2024)

    monkeypatch.setattr(config, "CORPUS_DIR", corpus)
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(retrieve, "_client", None)  # force fresh client at temp path
    build_index.main()
    return tmp_path


def test_metadata_filter_excludes_out_of_year_and_ranks_gold_first(tiny_index):
    # "March 2023" -> filter admits {2023, 2024} (year+1 lag) and EXCLUDES 2020,
    # even though 2020 is textually near-identical to the gold doc.
    hits = retrieve_engineered("What was the Japanese Yen position as of March 2023?")
    assert hits, "retrieval returned nothing"
    files = [h["source_file"] for h in hits]
    assert "treasury_bulletin_2020_03.txt" not in files, "filter must exclude 2020"
    assert hits[0]["source_file"] == "treasury_bulletin_2023_03.txt"
    assert hits[0]["year"] == "2023"


def test_full_pipeline_scorecard(tiny_index, monkeypatch):
    # Mock the LLM so the test is deterministic and offline.
    monkeypatch.setattr(generate, "ollama_generate",
                        lambda prompt, system="", **kw: "<FINAL_ANSWER>935851121560</FINAL_ANSWER>")

    eval_df = pd.DataFrame([{
        "uid": "T1",
        "question": "What was the Japanese Yen position as of March 2023?",
        "answer": "935851121560",
        "gold_files": ["treasury_bulletin_2023_03.txt"],
        "difficulty": "easy",
    }])

    result = evaluate_system("engineered", config.COLLECTIONS["engineered"],
                             eval_df, retrieve_engineered)
    sc = result["scorecard"]

    # retriever found the right bulletin AND the exact figure
    assert sc["doc"]["hit_rate@5"] == 1.0
    assert sc["cell"]["hit_rate@5"] == 1.0
    assert sc["doc"]["mrr"] == 1.0
    # generator: mocked answer matches gold within tolerance via official reward.py
    assert sc["factual_accuracy"] == 1.0
    # grounded: the asserted number is present in the retrieved context
    assert sc["groundedness"] == 1.0
    assert sc["hallucination_rate"] == 0.0
