"""
Central configuration for the RAG Groundedness Auditor.
Change model names, chunk sizes, and thresholds here — nowhere else.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # loads variables from a local .env file (never commit that file)

# --- Embedding model (Module 2) ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- NLI / groundedness verification model (Module 5) ---
NLI_MODEL = "facebook/bart-large-mnli"
ENTAILMENT_THRESHOLD = 0.5      # below this = not well-supported
CONTRADICTION_THRESHOLD = 0.4   # above this = flag as contradicted

# --- Generator LLM (Module 4) ---
# Choose ONE generation backend: "ollama" (local) or "nim" (NVIDIA NIM API)
GENERATION_BACKEND = "ollama"
OLLAMA_MODEL = "llama3.1:8b"

NIM_API_KEY = os.getenv("NVIDIA_NIM_API_KEY")
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_MODEL = "meta/llama-3.1-8b-instruct"

# --- Chunking (Module 1) ---
CHUNK_SIZE = 500      # words per chunk
CHUNK_OVERLAP = 50    # words of overlap between consecutive chunks

# --- Retrieval (Module 3) ---
TOP_K = 4

# --- Paths ---
DATA_DIR = "data"
RAW_FILINGS_DIR = f"{DATA_DIR}/raw_filings"
LABELED_TESTSET_PATH = f"{DATA_DIR}/labeled_testset.json"
FAISS_INDEX_DIR = "outputs/faiss_index"
EVAL_REPORT_PATH = "outputs/eval_report.json"