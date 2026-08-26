# RAG Groundedness Auditor

A retrieval-augmented QA system over financial filings (SEC 10-Ks/10-Qs) with an
automated hallucination-detection layer. Every generated answer is required to
cite its source, and each cited claim is independently checked against that
source using NLI-based entailment scoring — before the answer is considered
trustworthy.

## Why this project

Most RAG demos stop at "the LLM produced an answer." This project treats that
as the starting point, not the finish line: it adds a verification layer that
catches cases where the LLM's citation doesn't actually match what its source
says — the same class of problem GenAI governance tools solve for regulated,
high-stakes use cases.

## Pipeline

1. **Ingestion & Chunking** (`src/ingest.py`) — splits filings into
   metadata-tagged chunks.
2. **Embedding & Vector Store** (`src/embed_store.py`) — embeds chunks with
   `all-MiniLM-L6-v2` and indexes them with FAISS.
3. **Retrieval** (`src/retrieve.py`) — returns top-k relevant chunks for a
   query.
4. **Grounded Generation** (`src/generate.py`) — prompts an LLM (local via
   Ollama, or NVIDIA NIM's free API) to answer using only the retrieved
   sources, with inline citations.
5. **Groundedness Verification** (`src/verify.py`) — runs an NLI model
   (`facebook/bart-large-mnli`) to check whether each cited claim is actually
   entailed by its source.
6. **Evaluation & Reporting** (`src/evaluate.py`) — runs the full pipeline
   over a hand-labeled test set and reports precision/recall of the
   hallucination-flagging system.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your NVIDIA NIM key if using that backend

# For local generation instead, install Ollama and pull a model:
ollama pull llama3.1:8b
```

## Usage

```bash
# 1. Add plain-text filings to data/raw_filings/ (e.g. from SEC EDGAR)

# 2. Build the index
python -m src.ingest
python -m src.embed_store

# 3. Ask a question
python main.py

# 4. Run full evaluation (requires data/labeled_testset.json)
python -m src.evaluate
```

## Results

_To be filled in after evaluation: precision, recall, and example
transcripts from `outputs/eval_report.json`._

## Known limitations

- Citation parsing (`extract_citations`) is regex-based and may miss
  unconventional formatting.
- NLI model calibration on domain-specific financial text has not been
  independently verified against a general-purpose benchmark — this is
  exactly what the evaluation step in Module 6 is for.
