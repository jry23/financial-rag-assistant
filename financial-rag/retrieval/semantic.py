"""Semantic retrieval. Loads FAISS index + metadata lazily and exposes retrieve()."""
from __future__ import annotations
import json
from functools import lru_cache
from typing import List, Dict

import numpy as np
import faiss
from openai import OpenAI

from configs.config import (
    INDEX_PATH, CHUNKS_META_PATH, EMBED_MODEL, TOP_K,
)

client = OpenAI()


@lru_cache(maxsize=1)
def _load():
    index = faiss.read_index(str(INDEX_PATH))
    meta = json.loads(CHUNKS_META_PATH.read_text(encoding="utf-8"))
    return index, meta


def embed_query(q: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[q])
    v = np.array([resp.data[0].embedding], dtype="float32")
    faiss.normalize_L2(v)
    return v


def retrieve(query: str, k: int = TOP_K) -> List[Dict]:
    """Returns list of {score, chunk_id, section_name, text, ...} sorted by relevance."""
    index, meta = _load()
    qv = embed_query(query)
    scores, idxs = index.search(qv, k)
    results = []
    for score, i in zip(scores[0], idxs[0]):
        if i < 0:
            continue
        chunk = meta[i]
        results.append({
            "score": float(score),
            **chunk,
        })
    return results


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What are the main risk factors?"
    hits = retrieve(q)
    for h in hits:
        print(f"[{h['score']:.3f}] {h['chunk_id']} ({h['section_name']})")
        print(h['text'][:200].replace("\n", " "), "...\n")
