"""Validation report: trace every result against the answer key.

Confirms the scores are honest by showing, per question, whether the right
document was retrieved (doc_hit), whether the exact gold value was in the
retrieved chunks (cell_hit), and whether the graded answer was correct.

Writes results/validation.md.
"""
import json

import config
from src.common import load_eval_set


def load(system):
    return json.load(open(config.RESULTS_DIR / f"{system}.json"))["per_question"]


def main():
    df = load_eval_set().set_index("uid")
    lines = ["# Answer-key validation\n"]

    for system in ("baseline", "engineered"):
        pq = load(system)
        agg = {}
        for r in pq:
            d = df.loc[r["uid"], "difficulty"]
            a = agg.setdefault(d, [0, 0, 0, 0])
            a[0] += r["doc_hit"]
            a[1] += r["cell_hit"]
            a[2] += r["correct"]
            a[3] += 1
        summary = ", ".join(f"{d}: doc {v[0]}/{v[3]} · cell {v[1]}/{v[3]} · correct {v[2]}/{v[3]}"
                            for d, v in agg.items())
        lines.append(f"**{system}** — {summary}")

    lines.append("\n## Engineered per-question (does doc-hit find the exact figure?)\n")
    lines.append("| uid | difficulty | gold | doc_hit | cell_hit | correct |")
    lines.append("|---|---|---|---|---|---|")
    for r in load("engineered"):
        d = df.loc[r["uid"], "difficulty"]
        lines.append(f"| {r['uid']} | {d} | {r['gold_answer'][:20]} | "
                     f"{'yes' if r['doc_hit'] else 'no'} | {'yes' if r['cell_hit'] else 'NO'} | "
                     f"{'yes' if r['correct'] else 'no'} |")
    lines.append("\n_Doc-hit but not cell-hit ⇒ found the right bulletin, not the exact "
                 "cell (chunking-granularity limit). Many hard golds are derived "
                 "statistics never printed in the source ⇒ cell-hit impossible._")

    out = config.RESULTS_DIR / "validation.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
