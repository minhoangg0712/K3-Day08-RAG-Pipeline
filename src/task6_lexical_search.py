"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from .task4_chunking_indexing import load_documents


PROJECT_DIR = Path(__file__).resolve().parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"

# Cache corpus và index trong process hiện tại để tăng tốc truy vấn lặp lại.
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}
_LEXICAL_INDEX: dict[str, Any] | None = None


def _tokenize(text: str) -> list[str]:
    """Tokenize đơn giản, phù hợp cho tiếng Anh/Việt không dấu và ký hiệu thường gặp."""
    if not text:
        return []
    return re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)


def _ensure_corpus_loaded() -> list[dict]:
    """Nạp corpus markdown từ data/standardized nếu chưa có."""
    global CORPUS
    if CORPUS:
        return CORPUS

    loaded = load_documents(STANDARDIZED_DIR)
    CORPUS = [doc for doc in loaded if str(doc.get("content", "")).strip()]
    return CORPUS


def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """Chuẩn hóa score về [0, 1] để có thể phối hợp nhiều thang điểm."""
    if scores.size == 0:
        return scores
    min_v = float(np.min(scores))
    max_v = float(np.max(scores))
    if math.isclose(max_v, min_v):
        return np.zeros_like(scores, dtype=float)
    return (scores - min_v) / (max_v - min_v)


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    if not corpus:
        raise ValueError("Corpus rỗng, không thể build lexical index")

    tokenized_corpus = [_tokenize(str(doc.get("content", ""))) for doc in corpus]

    # BM25 là sparse ranker chính cho Task 6.
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi(tokenized_corpus)

    # TF-IDF được giữ song song để hỗ trợ trường hợp query rất ngắn/hiếm.
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf_vectorizer = TfidfVectorizer(tokenizer=_tokenize, token_pattern=None)
    tfidf_matrix = tfidf_vectorizer.fit_transform(
        [str(doc.get("content", "")) for doc in corpus]
    )

    return {
        "bm25": bm25,
        "tokenized_corpus": tokenized_corpus,
        "tfidf_vectorizer": tfidf_vectorizer,
        "tfidf_matrix": tfidf_matrix,
    }


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    global _LEXICAL_INDEX

    if not isinstance(query, str):
        raise TypeError("query phải là string")
    query = query.strip()
    if not query:
        return []

    if top_k <= 0:
        return []

    corpus = _ensure_corpus_loaded()
    if not corpus:
        return []

    if _LEXICAL_INDEX is None:
        _LEXICAL_INDEX = build_bm25_index(corpus)

    tokenized_query = _tokenize(query)
    if not tokenized_query:
        return []

    bm25 = _LEXICAL_INDEX["bm25"]
    tfidf_vectorizer = _LEXICAL_INDEX["tfidf_vectorizer"]
    tfidf_matrix = _LEXICAL_INDEX["tfidf_matrix"]

    # BM25 score (ranker chính)
    bm25_scores = np.asarray(bm25.get_scores(tokenized_query), dtype=float)

    # TF-IDF cosine similarity (booster lexical phụ)
    query_vec = tfidf_vectorizer.transform([query])
    tfidf_scores = (tfidf_matrix @ query_vec.T).toarray().ravel().astype(float)

    # Kết hợp: ưu tiên BM25, dùng TF-IDF để ổn định tie-break và truy vấn ngắn.
    combined_scores = 0.7 * _min_max_normalize(bm25_scores) + 0.3 * _min_max_normalize(
        tfidf_scores
    )

    top_k = min(top_k, len(corpus))
    top_indices = np.argsort(combined_scores)[::-1][:top_k]

    results: list[dict] = []
    for idx in top_indices:
        bm25_score = float(bm25_scores[idx])
        tfidf_score = float(tfidf_scores[idx])
        final_score = float(combined_scores[idx])

        # Giữ score dương theo yêu cầu test keyword-match; bỏ các kết quả hoàn toàn rỗng tín hiệu.
        if bm25_score <= 0.0 and tfidf_score <= 0.0:
            continue

        results.append(
            {
                "content": str(corpus[idx].get("content", "")),
                "score": round(final_score, 6),
                "metadata": dict(corpus[idx].get("metadata") or {}),
            }
        )

    # Đảm bảo sorted giảm dần theo score trả về.
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    # Test
    results = lexical_search("tuition fee payment methods", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
