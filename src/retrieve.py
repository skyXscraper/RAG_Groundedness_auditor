
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.embed_store import get_embedding_model, load_index


def retrieve(query: str, index, chunks, k: int = None):
    k=k or config.TOP_K
    model = get_embedding_model()

    q_embedding = model.encode([query], normalize_embeddings=True)
    scores, idxs = index.search(np.array(q_embedding, dtype="float32"), k)

    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue  
        results.append((chunks[idx], float(score)))

    return results


if __name__ == "__main__":
    index, chunks = load_index()
    query = input("Enter a test query: ")
    results = retrieve(query, index, chunks)

    print(f"\nTop {len(results)} results:\n")
    for chunk, score in results:
        print(f"[score={score:.3f}] ({chunk.source_id} / {chunk.section})")
        print(chunk.text[:200] + "...\n")