"""Embed chunks with OpenAI and build a FAISS index.

Uses cosine similarity via normalized vectors on an IndexFlatIP (inner product).
Fine up to ~100k chunks; swap for IndexHNSWFlat or a managed store later.
"""
from __future__ import annotations
import json
from typing import List

import numpy as np
import faiss
from openai import OpenAI
from tqdm import tqdm

from configs.config import (
    CHUNKS_PATH, INDEX_PATH, CHUNKS_META_PATH, EMBED_MODEL, EMBED_DIM,
)

BATCH = 100
client = OpenAI()


def embed_texts(texts: List[str]) -> np.ndarray:
    vecs = []
    for i in tqdm(range(0, len(texts), BATCH), desc="Embedding"):
        batch = texts[i: i + BATCH]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vecs.extend([d.embedding for d in resp.data])
    arr = np.array(vecs, dtype="float32")
    # L2-normalize so inner product == cosine similarity
    faiss.normalize_L2(arr)
    return arr


def main():
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(chunks)} chunks")

    texts = [c["text"] for c in chunks]
    vecs = embed_texts(texts)
    assert vecs.shape == (len(chunks), EMBED_DIM), f"got {vecs.shape}"

    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vecs)
    faiss.write_index(index, str(INDEX_PATH))
    print(f"Index: {index.ntotal} vectors -> {INDEX_PATH}")

    # metadata stays as JSON so we can inspect it; index row i ↔ chunks[i]
    CHUNKS_META_PATH.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    print(f"Metadata -> {CHUNKS_META_PATH}")


if __name__ == "__main__":
    main()
