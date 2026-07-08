"""Evaluate one system (baseline or engineered) → the 6-metric scorecard.

Set A — Retriever, reported at TWO relevance bars (both true, very different):
  DOC-level  : chunk comes from a gold `source_file` — "found the right bulletin?"
  CELL-level : chunk ALSO literally contains the gold value (reward.py number
               match) — "found the exact figure?" Brutal on OfficeQA because most
               answers are derived statistics never printed in the source.
  For each bar: Hit Rate@5, MRR, Recall@5.

Set B — Generator (deterministic, no LLM judge):
  Factual Accuracy   : official Databricks reward.score_answer, tolerance ±1%.
  Groundedness       : numeric claims in the answer that appear in the retrieved
                       context / total numeric claims (micro-averaged).
  Hallucination Rate : 1 − Groundedness.
"""
import json

import config
from reward import extract_final_answer, score_answer
from src.generate import generate_answer
from src.matching import answer_in_text, numeric_claims
from src.retrieve import all_chunks_from


def _hit_mrr(flags):
    hit = any(flags)
    return hit, (1.0 / (flags.index(True) + 1) if hit else 0.0)


def _cell_relevant_in_index(collection_name, gold_files, answer):
    n = 0
    for src in gold_files:
        for doc in all_chunks_from(collection_name, src):
            if answer_in_text(answer, doc):
                n += 1
    return n


def evaluate_system(name, collection_name, eval_df, retrieve_fn):
    acc = {lvl: {"hit": 0, "rr": [], "recall": []} for lvl in ("doc", "cell")}
    correct = 0
    total_claims = supported_claims = 0
    per_q = []

    for _, row in eval_df.iterrows():
        q, gold_ans, gold = row["question"], row["answer"], set(row["gold_files"])
        hits = retrieve_fn(q)
        ret_files = [h["source_file"] for h in hits]

        doc_flags = [f in gold for f in ret_files]
        cell_flags = [d and answer_in_text(gold_ans, h["text"])
                      for d, h in zip(doc_flags, hits)]
        for lvl, flags in (("doc", doc_flags), ("cell", cell_flags)):
            hit, rr = _hit_mrr(flags)
            acc[lvl]["hit"] += hit
            acc[lvl]["rr"].append(rr)
        found_docs = {f for f, d in zip(ret_files, doc_flags) if d}
        acc["doc"]["recall"].append(len(found_docs) / len(gold) if gold else 0.0)
        denom = _cell_relevant_in_index(collection_name, gold, gold_ans)
        acc["cell"]["recall"].append(sum(cell_flags) / denom if denom else 0.0)

        # --- generation ---
        answer = generate_answer(q, hits)
        is_correct = bool(score_answer(gold_ans, answer, config.FACTUAL_REL_TOLERANCE))
        correct += is_correct

        context = "\n".join(h["text"] for h in hits)
        final = extract_final_answer(answer)
        claims = numeric_claims(final)
        q_supported = sum(answer_in_text(c, context) for c in claims)
        total_claims += len(claims)
        supported_claims += q_supported

        per_q.append({
            "uid": row["uid"], "question": q, "gold_answer": gold_ans,
            "gold_files": sorted(gold), "retrieved": ret_files,
            "doc_hit": any(doc_flags), "cell_hit": any(cell_flags),
            "answer": answer, "final_answer": final, "correct": is_correct,
            "claims": len(claims), "supported": q_supported,
        })
        print(f"  [{name}] {row['uid']}: doc_hit={any(doc_flags):d} "
              f"cell_hit={any(cell_flags):d} correct={is_correct:d}")

    n = len(eval_df)
    grounded = supported_claims / total_claims if total_claims else None
    scorecard = {
        "system": name, "n_questions": n,
        "doc": {"hit_rate@5": round(acc["doc"]["hit"] / n, 4),
                "mrr": round(sum(acc["doc"]["rr"]) / n, 4),
                "recall@5": round(sum(acc["doc"]["recall"]) / n, 4)},
        "cell": {"hit_rate@5": round(acc["cell"]["hit"] / n, 4),
                 "mrr": round(sum(acc["cell"]["rr"]) / n, 4),
                 "recall@5": round(sum(acc["cell"]["recall"]) / n, 4)},
        "factual_accuracy": round(correct / n, 4),
        "groundedness": round(grounded, 4) if grounded is not None else None,
        "hallucination_rate": round(1 - grounded, 4) if grounded is not None else None,
        "claims_checked": total_claims,
    }
    return {"scorecard": scorecard, "per_question": per_q}


def save(result, path):
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
