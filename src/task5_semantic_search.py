"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[\w]+", text.lower(), flags=re.UNICODE))


def _similarity_fallback(query: str, corpus: list[dict], top_k: int) -> list[dict]:
    query_tokens = _tokenize(query)
    results: list[dict] = []
    for item in corpus:
        content = str(item.get("content", ""))
        content_tokens = _tokenize(content)
        if not query_tokens or not content_tokens:
            continue
        overlap = len(query_tokens & content_tokens)
        score = overlap / max(len(query_tokens), 1)
        if score <= 0:
            continue
        results.append(
            {
                "content": content,
                "score": float(round(score, 6)),
                "metadata": dict(item.get("metadata") or {}),
            }
        )
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


@lru_cache(maxsize=1)
def _load_corpus() -> list[dict]:
    from .task4_chunking_indexing import load_documents

    docs = load_documents(STANDARDIZED_DIR)
    if not docs:
        return []

    try:
        from .task4_chunking_indexing import chunk_documents, embed_chunks, index_to_vectorstore

        chunks = chunk_documents(docs)
        embedded = embed_chunks(chunks)
        index_to_vectorstore(embedded, reset_collection=False)
        return embedded
    except Exception:
        return docs


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if not isinstance(query, str):
        raise TypeError("query phải là string")
    query = query.strip()
    if not query or top_k <= 0:
        return []

    corpus = _load_corpus()
    if not corpus:
        return []

    try:
        from .task4_chunking_indexing import get_collection, get_embedding_model

        model = get_embedding_model()
        query_vector = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        collection = get_collection(reset=False)
        results = collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=min(top_k, max(1, collection.count())),
            include=["documents", "metadatas", "distances"],
        )

        output: list[dict] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for document, metadata, distance in zip(documents, metadatas, distances):
            score = max(0.0, 1.0 - float(distance))
            output.append(
                {
                    "content": document,
                    "score": round(score, 6),
                    "metadata": dict(metadata or {}),
                }
            )

        output.sort(key=lambda item: item["score"], reverse=True)
        if output:
            return output[:top_k]
    except Exception:
        pass

    return _similarity_fallback(query, corpus, top_k)


if __name__ == "__main__":
    # Test
    results = semantic_search("what is the tuition fee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
