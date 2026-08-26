"""
Runs the full RAG Groundedness Auditor pipeline end to end for a single query.
For building the index from scratch, run src/ingest.py then src/embed_store.py first.
For full evaluation over a labeled test set, run src/evaluate.py instead.
"""

from src.embed_store import load_index
from src.retrieve import retrieve
from src.generate import build_prompt, generate_answer, extract_citations
from src.verify import score_answer
import config


def run_pipeline(query: str):
    index, chunks = load_index()

    retrieved = retrieve(query, index, chunks, k=config.TOP_K)
    prompt = build_prompt(query, retrieved)
    answer = generate_answer(prompt)
    sentences, citation_map = extract_citations(answer, len(retrieved))
    flagged = score_answer(sentences, citation_map, retrieved)

    return {
        "query": query,
        "answer": answer,
        "flagged_claims": flagged,
        "num_sources_used": len(retrieved),
    }


if __name__ == "__main__":
    query = input("Ask a question about the ingested filings: ")
    result = run_pipeline(query)

    print("\n=== ANSWER ===")
    print(result["answer"])

    print(f"\n=== GROUNDEDNESS CHECK ({len(result['flagged_claims'])} issues found) ===")
    if not result["flagged_claims"]:
        print("All cited claims appear well-supported by their sources.")
    else:
        for issue in result["flagged_claims"]:
            print(f"\n⚠ {issue['reason']}")
            print(f"  Claim: {issue['sentence']}")
            print(f"  Source: {issue['source_id']}")