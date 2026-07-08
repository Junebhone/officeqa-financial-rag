# Discussion Board Submission

**Name:** Bhone Kyaw Si Hein  |  **Recent Years Used:** 2015–2025 (quarterly Treasury Bulletins)
**GitHub:** https://github.com/Junebhone/officeqa-financial-rag

> **Scope note.** OfficeQA questions span the whole 1939–2025 archive and recent years
> are sparse — 2022–2025 alone yields only **3** fully-answerable questions. I widened
> to **2015–2025** for **17** evaluable questions (11 hard, 6 easy) so the percentages
> are stable. Generator: `qwen2.5:7b-instruct` (local Ollama). Factual accuracy uses
> Databricks' official `reward.py` (±1%).

## Part 1: The Scorecard (K=5)

| Metric | Baseline (Simple) | Engineered (Improved) |
|---|---|---|
| Hit Rate (K=5) | 11.8% | **58.8%** |
| MRR | 0.05 | **0.27** |
| Recall@5 | 5.9% | **47.1%** |
| Groundedness | 40.0%* | 10.0%* |
| Factual Accuracy | 0.0% | **5.9%** |
| Hallucination Rate | 60.0%* | 90.0%* |

Hit Rate / MRR / Recall are **document-level** (did we retrieve the right bulletin).
I also report a stricter **cell-level** bar (did a retrieved chunk contain the exact
gold figure): **0% for both systems** — see below.

\* Groundedness/Hallucination rest on very few asserted numbers (baseline 5, engineered
10 — the baseline mostly refuses, so it asserts fewer numbers). Treat these as noisy;
the real signal is in the retrieval metrics.

## Part 2: Engineering Reflection

**1. The Bottleneck — Finding or Understanding?**
**Finding the data (the Retriever).** The baseline's Hit Rate@5 = 11.8% and Recall@5 =
5.9% mean it retrieved the correct bulletin for only 2 of 17 questions. Factual accuracy
is capped by retrieval — the model can't read a table it never received — so the
retriever is the binding constraint. The metric that proves it is **Hit Rate@5**, not
Groundedness.

**2. The Metadata Fix — Retrieval or Generation?**
It helped the **Retriever dramatically, the Generator barely.** Year/Month metadata
pre-filtering (+ a year+1 fiscal-lag rule, structure-aware chunking, and a stronger
bge-small embedder) lifted **Hit Rate 11.8% → 58.8% (5×), MRR 0.05 → 0.27, Recall
5.9% → 47.1% (8×)**, while generation barely moved (**Factual Accuracy 0% → 5.9%**,
**Cell Hit Rate stayed 0%**). Metadata changes *which* documents are searched, not
whether the exact numeric cell lands in the top-5 chunks or whether the model can
*compute* a derived answer. Publication lag matters: "CY 2022" totals print in the
**March 2023** bulletin, so the filter includes year+1 (without it, one question's gold
doc was filtered out entirely).

**3. Scaling Insight — 4-year subset → 80-year archive (1939–2025)?**
**Retrieval precision on near-duplicate periodic tables breaks first.** The Bulletin
republishes identical table layouts every quarter, so 80 years is ~340 copies of, e.g.,
"Japanese Yen Positions" — vector similarity alone can't tell 1962-Q2 from 2019-Q2,
which is exactly why the Year/Month metadata filter is load-bearing and must be promoted
to **physical sharding** (partition the index by decade). Secondary breakages: the flat
HNSW index + "re-embed the whole corpus on every build" step become the throughput
bottleneck, and the modern chunker/prompt heuristics silently degrade on 1939-era
layouts and terminology.

---

*Reproduce:* `pip install -r requirements.txt`, `ollama pull qwen2.5:7b-instruct`,
`export HF_TOKEN=…` (gated dataset), `python src/download_data.py`, `python run.py`.
Tests (no data/Ollama needed): `pytest -q`.
