"""
Module 2: Embedding & Vector Store

Turns chunks into vectors and stores them in a FAISS index so they can
be searched by semantic similarity, not just keyword match.
"""

import os
import sys
import json
import pickle

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.ingest import Chunk

_model = None  # lazy-loaded singleton so we don't reload the model repeatedly


def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def build_index(chunks: list[Chunk]):
    """Embeds all chunks and builds a FAISS index (cosine similarity via inner product)."""
    model = get_embedding_model()
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings, dtype="float32"))

    return index, chunks


def save_index(index, chunks: list[Chunk], out_dir: str = None):
    out_dir = out_dir or config.FAISS_INDEX_DIR
    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(out_dir, "index.faiss"))
    with open(os.path.join(out_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)


def load_index(in_dir: str = None):
    in_dir = in_dir or config.FAISS_INDEX_DIR
    index = faiss.read_index(os.path.join(in_dir, "index.faiss"))
    with open(os.path.join(in_dir, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


if __name__ == "__main__":
    with open("data/chunks.json") as f:
        raw = json.load(f)
    chunks = [Chunk(**c) for c in raw]

    index, chunks = build_index(chunks)
    save_index(index, chunks)
    print(f"Indexed {len(chunks)} chunks -> saved to {config.FAISS_INDEX_DIR}")
    0