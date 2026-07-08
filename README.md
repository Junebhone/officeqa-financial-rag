# OfficeQA Financial RAG Challenge

A retrieval-augmented generation system that answers questions over **U.S. Treasury
Bulletin** records (Databricks [OfficeQA](https://huggingface.co/datasets/databricks/officeqa)
benchmark). It compares a **Baseline** (simple, unoptimized) pipeline against an
**Engineered** (improved) one across six retrieval + generation metrics.

- **Recent years used:** 2015–2025 (quarterly bulletins: Mar/Jun/Sep/Dec)
- **Evaluable questions:** 17 (11 hard, 6 easy) — every question whose gold source
  documents all fall inside the indexed corpus
- **Everything runs locally, no API key** (generation via Ollama)

---

## 1. Architecture

```
officeqa_full.csv ──(filter to in-scope Qs)──► eval set (17 Q/A + gold source_files)
                                                        │
treasury_bulletin_YYYY_MM.txt ──► chunk ──► embed ──► ChromaDB (2 collections)
                                                        │            tags: year, month, source_file
                                            question ──►│
                                                        ▼
                       retrieve top-5 ──► qwen2.5:7b answer ──► deterministic scoring
                                                        │            (reward.py + matching)
                                                        ▼
                                     Hit@5 · MRR · Recall@5 · Groundedness ·
                                     Factual Accuracy · Hallucination Rate
```

| Component | Choice |
|---|---|
| **Vector database** | ChromaDB (persistent, cosine space) |
| **Embeddings (baseline)** | `all-MiniLM-L6-v2` (384-dim) |
| **Embeddings (engineered)** | `BAAI/bge-small-en-v1.5` (384-dim) + query instruction |
| **Generator** | `qwen2.5:7b-instruct` via Ollama (temp 0), `<FINAL_ANSWER>` tagged |
| **Factual-accuracy scorer** | Databricks' official `reward.py` (`score_answer`, ±1%) |
| **Metadata (required)** | `year`, `month`, `source_file` on every chunk |
| **Retrieval cutoff** | K = 5 |

### Baseline vs Engineered

| Dimension | Baseline (Simple) | Engineered (Improved) |
|---|---|---|
| **Chunking** | Blind fixed **1000-char** windows, **no overlap** — cuts tables mid-row | **Structure-aware**: whole pipe-table rows kept together, packed to ~2000 chars (~512 tokens) with 300-char overlap, **section header prepended** for context |
| **Embedding** | MiniLM | bge-small (+ retrieval instruction on the query) |
| **Metadata filtering** | None — searches the whole corpus | **Year/Month `where` pre-filter** (+ **year+1 fiscal lag**): the question's years and any named month/quarter restrict the search to the right bulletins before ranking |

### How metadata is used (required write-up)
Each chunk is tagged `{year, month, source_file}` at index time. At query time the
engineered retriever parses **4-digit years** (`2015`–`2025`) and any **month name /
`Qn`** from the question, then passes a Chroma `where` clause
(`{"year": {"$in": [...]}, "month": "03"}`). This shrinks the candidate pool to the
correct quarterly bulletin(s) *before* vector ranking, so the top-5 isn't polluted by
identically-worded tables from other years (the Treasury Bulletin repeats the same
table layouts every quarter). Crucially the filter also includes **year+1**: the
Treasury reports year-end / fiscal-year data in the *following* bulletin (e.g. "CY
2022" totals appear in the **March 2023** issue), so a naive year lock would exclude
the true gold document. A safety net falls back to unfiltered search if a filter
returns nothing.

---

## 2. Metric definitions

**Retriever** — reported at **two relevance bars** (both scored vs the answer key's
`source_files`, K=5):
- **Doc-level** — a chunk is relevant if it comes from a gold document ("found the
  right *bulletin*?"). Hit Rate@5, MRR, and Recall@5 (distinct gold docs found / #
  gold docs). This is where the metadata filter shows up.
- **Cell-level** — stricter: the chunk must *also* literally contain the gold value
  (via `reward.py` number matching) — "found the exact *figure*?". Brutal on
  OfficeQA because most answers are derived statistics never printed in the source.

**Generator** (deterministic — no LLM judge, fully reproducible):
- **Factual Accuracy** — Databricks' official `reward.score_answer` (extracts the
  `<FINAL_ANSWER>` tag, unit-aware, ±1% tolerance).
- **Groundedness** — `numeric claims in the answer that appear in the retrieved
  context / total numeric claims` (micro-averaged, `reward.py` number matching).
- **Hallucination Rate** — `1 − Groundedness`.

---

## 3. Scorecard (K=5)

Generator: `qwen2.5:7b-instruct` (Ollama) · Factual accuracy: official `reward.py`
(±1%) · 17 questions (11 hard, 6 easy)

**Set A — Retriever** (Doc = found the right bulletin; Cell = found the exact figure)

| Metric | Baseline | Engineered |
|---|---|---|
| **Doc** Hit Rate@5 | 11.8% | **58.8%** |
| **Doc** MRR | 0.05 | **0.27** |
| **Doc** Recall@5 | 5.9% | **47.1%** |
| **Cell** Hit / MRR / Recall@5 | 0% / 0.00 / 0% | 0% / 0.00 / 0% |

**Set B — Generator**

| Metric | Baseline | Engineered |
|---|---|---|
| Factual Accuracy | 0.0% | **5.9%** (1/17) |
| Groundedness | 40.0%¹ | 10.0%¹ |
| Hallucination Rate | 60.0%¹ | 90.0%¹ |
| (numeric claims checked) | 5 | 10 |

¹ Groundedness rests on very few claims (baseline 5, engineered 10 — the baseline
mostly refuses, so it *asserts* fewer numbers), so these figures are noisy; the
signal is in **Set A**.

**Retrieval by difficulty (docs found in top-5):** baseline hard 2/11, easy 0/6 →
engineered hard **7/11**, easy **3/6**.

### Why Factual Accuracy is still near-zero (validated, not a bug)
Traced every question against the answer key (`results/validation.md`). Two structural
reasons, both real:
1. **Doc-level hit ≠ cell-level hit.** The engineered system retrieves the correct
   *bulletin* for 10/17 questions, but **Cell Hit Rate is 0%** — the exact gold figure
   is never in the top-5 chunks. Retrieval got the book; it didn't land on the page.
2. **Derived-answer questions.** Most hard golds (Zipf/Pareto exponents, QoQ
   differences) are **computed statistics never printed** in the source — a plain
   retrieve-then-generate pipeline cannot produce them. The one correct answer
   (UID0086, `4.815`) is the exception that proves the rule.
   (OfficeQA is a *grounded-reasoning* benchmark; even frontier agents score modestly.)

---

## 4. Setup & reproduction

```bash
pip install -r requirements.txt
ollama pull qwen2.5:7b-instruct

# The corpus + answer key are GATED on Hugging Face — request access, then:
export HF_TOKEN=hf_xxx
python src/download_data.py          # downloads 2015–2025 txt + officeqa_full.csv

python run.py                        # build indexes + evaluate both systems
python run.py --eval                 # re-evaluate without rebuilding indexes
```

Outputs land in `results/`: `scorecard.md`, `baseline.json`, `engineered.json`
(per-question doc/cell hits + answers + scores), and `validation.md`
(`python -m src.diagnostics`). The per-question JSON/validation files are gitignored
(gated benchmark content); only the aggregate `scorecard.md` is committed.

### Tests
```bash
pytest -q            # 11 tests, no gated data or Ollama needed (synthetic + mocked LLM)
```

### Layout
```
config.py              all knobs (years, chunk sizes, models, K)
reward.py              Databricks' official scorer (Apache-2.0), vendored
src/common.py          filename→year/month parsing, answer-key filter, Ollama client
src/chunking.py        baseline (fixed) + engineered (structure-aware, ALL-CAPS sections)
src/build_index.py     embed + store into two Chroma collections with metadata
src/retrieve.py        baseline top-k vs engineered year/month(+year+1) pre-filter
src/matching.py        number-containment (cell relevance + groundedness), on reward.py
src/generate.py        RAG answer generation (<FINAL_ANSWER> tagged)
src/evaluate.py        doc+cell retrieval metrics, reward.py accuracy, groundedness
src/diagnostics.py     answer-key validation report
run.py                 orchestrates the whole comparison
tests/                 unit + end-to-end tests
```

---

## 5. Engineering reflection

**1. The Bottleneck — Finding or Understanding?**
**Finding the data (the Retriever).** The baseline's `Doc Hit Rate@5 = 11.8%` and
`Recall@5 = 5.9%` mean it surfaced the correct bulletin for only 2 of 17 questions.
Factual Accuracy is capped by retrieval — the model can't read a table it never
received — so the retriever is the binding constraint, and the metric that proves it is
**Hit Rate@5 (retriever)**, not Groundedness. (The baseline's 40% Groundedness looks
higher than the engineered's 10%, but that's an artifact: the baseline asserts only 5
numbers total because it mostly refuses, whereas the engineered model — given real
context — attempts 10, most of which aren't verbatim in the chunks.)

**2. The Metadata Fix — did it help Retrieval or Generation more?**
It helped the **Retriever dramatically; the Generator barely.** Year/Month metadata
pre-filtering (+ year+1 fiscal lag, structure-aware chunking, and a stronger bge-small
embedder) moved the retrieval metrics: **Doc Hit Rate 11.8% → 58.8% (5×), MRR
0.05 → 0.27 (5.4×), Recall 5.9% → 47.1% (8×)**. Generation barely moved: **Factual
Accuracy 0% → 5.9%** (one derived answer), and **Cell Hit Rate stayed 0%**. That's the
whole lesson: the metadata filter changes *which documents* are searched — it does
nothing about whether the exact numeric cell lands in the top-5 chunks or whether the
model can *compute* a derived answer. Handling **publication lag** matters too: "CY
2022" totals are printed in the **March 2023** bulletin, so the filter includes year+1
(without it, UID0081's gold doc was filtered out entirely).

**3. Scaling Insight — what breaks first going from these years to 80 (1939–2025)?**
The **retrieval precision on near-duplicate periodic tables** breaks first. The Treasury
Bulletin republishes the *same* table layouts every quarter, so 80 years is ~340 copies
of, say, "Japanese Yen Positions." Vector similarity alone cannot distinguish 1962-Q2
from 2019-Q2 — which is exactly why the Year/Month metadata filter is load-bearing and
must be promoted to **physical sharding** (partition the index by decade) to keep
precision and latency sane. Even with a perfect year filter the weak link is already
visible: **Cell Hit Rate is 0%** — the right bulletin is found but the exact cell isn't
in the top-5 chunks — and at ~700 docs / 200k+ chunks that only worsens, while the naive
"re-embed the whole corpus into one flat HNSW index on every build" step becomes the
throughput bottleneck. The fix direction is **hierarchical retrieval** (metadata → doc →
section → cell) plus agentic/reasoning generation for the derived-statistic questions —
not a bigger flat top-5.
