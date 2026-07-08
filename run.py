"""End-to-end driver: build indexes, evaluate baseline + engineered, write scorecard.

Usage:
    python run.py            # full pipeline (build indexes + evaluate both)
    python run.py --eval     # skip index build (reuse chroma_db), just evaluate
"""
import sys

import config
from src import build_index
from src.common import load_eval_set
from src.evaluate import evaluate_system, save
from src.retrieve import retrieve_baseline, retrieve_engineered


def pct(x):
    return f"{x*100:.1f}%" if x is not None else "n/a"


def write_scorecard(base, eng, n):
    md = f"""# OfficeQA Financial RAG — Scorecard (K={config.TOP_K})

Corpus years: {config.YEARS[0]}-{config.YEARS[-1]} | Evaluable questions: {n}
Generator: {config.GEN_MODEL} (Ollama) | Factual accuracy: official reward.py (±1%)
Baseline embed: {config.BASELINE_EMBED_MODEL}
Engineered embed: {config.ENGINEERED_EMBED_MODEL}

### Set A — Retriever
| Metric | Baseline | Engineered |
|---|---|---|
| **Doc** Hit Rate@5 | {pct(base['doc']['hit_rate@5'])} | {pct(eng['doc']['hit_rate@5'])} |
| **Doc** MRR | {base['doc']['mrr']:.2f} | {eng['doc']['mrr']:.2f} |
| **Doc** Recall@5 | {pct(base['doc']['recall@5'])} | {pct(eng['doc']['recall@5'])} |
| **Cell** Hit Rate@5 | {pct(base['cell']['hit_rate@5'])} | {pct(eng['cell']['hit_rate@5'])} |
| **Cell** MRR | {base['cell']['mrr']:.2f} | {eng['cell']['mrr']:.2f} |
| **Cell** Recall@5 | {pct(base['cell']['recall@5'])} | {pct(eng['cell']['recall@5'])} |

### Set B — Generator
| Metric | Baseline | Engineered |
|---|---|---|
| Factual Accuracy | {pct(base['factual_accuracy'])} | {pct(eng['factual_accuracy'])} |
| Groundedness | {pct(base['groundedness'])} | {pct(eng['groundedness'])} |
| Hallucination Rate | {pct(base['hallucination_rate'])} | {pct(eng['hallucination_rate'])} |
| (claims checked) | {base['claims_checked']} | {eng['claims_checked']} |

_Doc-level = found the right bulletin; Cell-level = found the exact printed figure._
_Baseline: fixed 1000-char chunks, MiniLM, no metadata filter._
_Engineered: structure-aware chunks (tables intact + section context), bge-small, Year/Month metadata pre-filter (+year+1 fiscal lag)._
"""
    path = config.RESULTS_DIR / "scorecard.md"
    path.write_text(md)
    print("\n" + md)
    print(f"Wrote {path}")


def main():
    config.RESULTS_DIR.mkdir(exist_ok=True)
    if "--eval" not in sys.argv:
        build_index.main()

    eval_df = load_eval_set()
    print(f"\nEvaluating on {len(eval_df)} questions "
          f"({eval_df['difficulty'].value_counts().to_dict()})\n")

    print("=== BASELINE ===")
    base = evaluate_system("baseline", config.COLLECTIONS["baseline"],
                           eval_df, retrieve_baseline)
    save(base, config.RESULTS_DIR / "baseline.json")

    print("\n=== ENGINEERED ===")
    eng = evaluate_system("engineered", config.COLLECTIONS["engineered"],
                          eval_df, retrieve_engineered)
    save(eng, config.RESULTS_DIR / "engineered.json")

    write_scorecard(base["scorecard"], eng["scorecard"], len(eval_df))


if __name__ == "__main__":
    main()
