"""Central configuration for the OfficeQA Financial RAG lab.

Two systems are defined and compared:
  - BASELINE  : naive fixed-size character chunking, small embedding model,
                NO metadata filtering. The "simple librarian".
  - ENGINEERED: structure-aware chunking (tables kept whole + section context),
                stronger embedding model, Year/Month metadata pre-filtering
                driven by the question. The "improved librarian".

Everything a grader needs to reproduce the run lives here.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"                 # treasury_bulletin_YYYY_MM.txt files
ANSWER_KEY_CSV = DATA_DIR / "officeqa_full.csv"  # gated benchmark CSV
CHROMA_DIR = ROOT / "chroma_db"
RESULTS_DIR = ROOT / "results"

# ---- Scope: which years form the corpus + eval set --------------------------
# Chosen for the lab: 2015-2025 -> ~43 quarterly docs, 17 evaluable questions.
YEARS = [str(y) for y in range(2015, 2026)]

# ---- Retrieval cutoff -------------------------------------------------------
TOP_K = 5

# ---- Embedding models (downloaded automatically by sentence-transformers) ---
# Baseline uses a small, fast model; engineered uses a stronger retrieval model.
BASELINE_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384-dim
ENGINEERED_EMBED_MODEL = "BAAI/bge-small-en-v1.5"                 # 384-dim, better recall
# bge models expect a short instruction prefixed to the *query* (not the docs):
ENGINEERED_QUERY_PREFIX = "Represent this financial question for retrieving supporting passages: "

# ---- Chunking ---------------------------------------------------------------
# Baseline: blind fixed-width character windows, NO overlap -> tables get cut.
BASELINE_CHUNK_CHARS = 1000
BASELINE_CHUNK_OVERLAP = 0

# Engineered: pack structural blocks (paragraphs / whole table rows) up to a
# target size, with overlap, and prepend the nearest section header for context.
# ~512 tokens ~= 2000 chars; overlap ~15%.
ENGINEERED_CHUNK_CHARS = 2000
ENGINEERED_CHUNK_OVERLAP = 300

# ---- Generation / judging (local Ollama, no API key) ------------------------
GEN_PROVIDER = "ollama"                     # "ollama" | "anthropic" | "openai"
GEN_MODEL = "qwen2.5:7b-instruct"
OLLAMA_URL = "http://localhost:11434"
GEN_TEMPERATURE = 0.0                       # deterministic for reproducible metrics
GEN_NUM_CTX = 8192

# Numeric factual-accuracy tolerance (assignment: "match the CSV +/- 1%").
FACTUAL_REL_TOLERANCE = 0.01

COLLECTIONS = {
    "baseline": "officeqa_baseline",
    "engineered": "officeqa_engineered",
}
