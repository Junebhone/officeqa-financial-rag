# OfficeQA Financial RAG — Scorecard (K=5)

Corpus years: 2015-2025 | Evaluable questions: 17
Generator: qwen2.5:7b-instruct (Ollama) | Factual accuracy: official reward.py (±1%)
Baseline embed: sentence-transformers/all-MiniLM-L6-v2
Engineered embed: BAAI/bge-small-en-v1.5

### Set A — Retriever
| Metric | Baseline | Engineered |
|---|---|---|
| **Doc** Hit Rate@5 | 11.8% | 58.8% |
| **Doc** MRR | 0.05 | 0.27 |
| **Doc** Recall@5 | 5.9% | 47.1% |
| **Cell** Hit Rate@5 | 0.0% | 0.0% |
| **Cell** MRR | 0.00 | 0.00 |
| **Cell** Recall@5 | 0.0% | 0.0% |

### Set B — Generator
| Metric | Baseline | Engineered |
|---|---|---|
| Factual Accuracy | 0.0% | 5.9% |
| Groundedness | 40.0% | 10.0% |
| Hallucination Rate | 60.0% | 90.0% |
| (claims checked) | 5 | 10 |

_Doc-level = found the right bulletin; Cell-level = found the exact printed figure._
_Baseline: fixed 1000-char chunks, MiniLM, no metadata filter._
_Engineered: structure-aware chunks (tables intact + section context), bge-small, Year/Month metadata pre-filter (+year+1 fiscal lag)._
