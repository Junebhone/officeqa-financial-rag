"""RAG answer generation (Ollama).

Answers end with a <FINAL_ANSWER>...</FINAL_ANSWER> tag so the official
reward.py scorer can extract a clean, direct answer. Grounding and factual
accuracy are scored deterministically in evaluate.py (no LLM judge), so this
module only has to produce answers.
"""
from src.common import ollama_generate

ANSWER_SYSTEM = (
    "You are a careful financial analyst answering questions from U.S. Treasury "
    "Bulletin excerpts. Use ONLY numbers that literally appear in the context; "
    "never estimate. Quote units exactly as printed (millions vs billions "
    "matters). If the context does not contain the answer, the final answer is "
    "exactly NOT FOUND."
)

ANSWER_TEMPLATE = """Context passages:
{context}

Question: {question}

Think step by step about which table and row applies, then end your reply with
the exact answer inside tags, e.g. <FINAL_ANSWER>1234.5 million</FINAL_ANSWER>."""


def generate_answer(question: str, hits: list[dict]) -> str:
    context = "\n\n---\n\n".join(f"[{h['source_file']}] {h['text']}" for h in hits)
    prompt = ANSWER_TEMPLATE.format(context=context[:12000], question=question)
    return ollama_generate(prompt, system=ANSWER_SYSTEM)
