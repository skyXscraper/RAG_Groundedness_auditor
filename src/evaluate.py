"""
Module 6: Evaluation & Reporting

Runs the full pipeline (retrieve -> generate -> verify) over a hand-labeled
test set and reports precision/recall of the hallucination-flagging system,
plus example transcripts for your write-up.

Expected labeled_testset.json format:
[
  {"question": "...", "is_hallucination_expected": true/false},
  ...
]
"is_hallucination_expected" = true means you, the human labeler, believe
a correct/careful system SHOULD flag something in the retrieved context
for this question (e.g. it's an adversarial or unanswerable question).
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.embed_store import load_index
from src.retrieve import retrieve
from src.generate import build_prompt, generate_answer, extract_citations
from src.verify import score_answer


def run_single(query: str, index, chunks) -> dict:
    retrieved = retrieve(query, index, chunks)
    prompt = build_prompt(query, retrieved)
    answer = generate_answer(prompt)
    sentences, citation_map = extract_citations(answer, len(retrieved))
    flagged = score_answer(sentences, citation_map, retrieved)

    return {
        "question": query,
        "answer": answer,
        "flagged": flagged,
        "predicted_hallucination": len(flagged) > 0,
    }


def evaluate(testset_path: str = None) -> dict:
    testset_path = testset_path or config.LABELED_TESTSET_PATH
    with open(testset_path, "r", encoding="utf-8") as f:
        testset = json.load(f)

    index, chunks = load_index()

    tp = fp = tn = fn = 0
    transcripts = []

    for item in testset:
        result = run_single(item["question"], index, chunks)
        predicted = result["predicted_hallucination"]
        actual = item["is_hallucination_expected"]

        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1

        transcripts.append({**result, "actual_hallucination_expected": actual})

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    report = {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "transcripts": transcripts,
    }

    os.makedirs(os.path.dirname(config.EVAL_REPORT_PATH), exist_ok=True)
    with open(config.EVAL_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    report = evaluate()
    print(f"Precision: {report['precision']}")
    print(f"Recall:    {report['recall']}")
    print(f"F1:        {report['f1']}")
    print(f"TP={report['tp']} FP={report['fp']} TN={report['tn']} FN={report['fn']}")
    print(f"\nFull report saved to {config.EVAL_REPORT_PATH}")