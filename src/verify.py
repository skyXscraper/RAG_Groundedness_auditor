"""
Module 5: Groundedness Verification

The core differentiator of this project. Independently checks whether
each cited claim in the generated answer is actually entailed by the
source it cites, using an NLI (entailment) model — rather than trusting
the LLM's own citation.
"""

import os
import sys

from transformers import pipeline

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

_nli = None  # lazy-loaded singleton


def get_nli_model():
    global _nli
    if _nli is None:
        _nli = pipeline("text-classification", model=config.NLI_MODEL, top_k=None)
    return _nli


def check_claim(claim: str, source_text: str) -> dict:
    """Returns {'ENTAILMENT': score, 'NEUTRAL': score, 'CONTRADICTION': score}."""
    nli = get_nli_model()
    # bart-large-mnli expects (premise, hypothesis) — source is the premise,
    # the generated claim is the hypothesis we're checking.
    result = nli(source_text, text_pair=claim)
    return {r["label"].upper(): r["score"] for r in result[0]} if isinstance(result[0], list) \
        else {r["label"].upper(): r["score"] for r in result}


def score_answer(sentences: list, citation_map: dict, retrieved_chunks: list) -> list:
    """
    Checks every cited sentence against its source.
    Returns a list of flagged issues: [{sentence, source_id, scores, reason}, ...]
    Unflagged sentences are considered adequately grounded.
    """
    flagged = []

    for sent_idx, src_idx in citation_map.items():
        claim = sentences[sent_idx]
        chunk, _ = retrieved_chunks[src_idx]
        scores = check_claim(claim, chunk.text)

        entailment = scores.get("ENTAILMENT", 0)
        contradiction = scores.get("CONTRADICTION", 0)

        reason = None
        if contradiction > config.CONTRADICTION_THRESHOLD:
            reason = "contradicted by source"
        elif entailment < config.ENTAILMENT_THRESHOLD:
            reason = "not clearly supported by source"

        if reason:
            flagged.append({
                "sentence": claim,
                "source_id": chunk.source_id,
                "scores": scores,
                "reason": reason,
            })

    return flagged


if __name__ == "__main__":
    from src.embed_store import load_index
    from src.retrieve import retrieve
    from src.generate import build_prompt, generate_answer, extract_citations

    index, chunks = load_index()
    query = input("Enter a test query: ")
    retrieved = retrieve(query, index, chunks)

    prompt = build_prompt(query, retrieved)
    answer = generate_answer(prompt)
    sentences, citation_map = extract_citations(answer, len(retrieved))

    flagged = score_answer(sentences, citation_map, retrieved)

    print("\n--- Answer ---")
    print(answer)
    print(f"\n--- Flagged claims ({len(flagged)}) ---")
    for item in flagged:
        print(f"\n⚠ {item['reason'].upper()}")
        print(f"Claim: {item['sentence']}")
        print(f"Source: {item['source_id']}")
        print(f"Scores: {item['scores']}")