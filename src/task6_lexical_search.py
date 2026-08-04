"""
Task 6 — Lexical Search Module (Sparse Retrieval).

Cài đặt 2 phương pháp để so sánh trong demo (tiêu chí bonus +5đ):

    BM25 (mặc định) — rank_bm25.BM25Okapi
        score(q,d) = Σ IDF(qi) · tf(qi,d)·(k1+1) / (tf(qi,d) + k1·(1-b+b·|d|/avgdl))
        Hai cơ chế mà TF-IDF không có:
          • Bão hoà tần suất (k1=1.5): từ khoá xuất hiện lần thứ 10 gần như không
            cộng thêm điểm so với lần thứ 9. TF-IDF thì cộng tuyến tính vô hạn,
            nên một chunk lặp "người lao động" 20 lần sẽ thắng oan.
          • Chuẩn hoá độ dài (b=0.75): phạt document dài theo |d|/avgdl. Quan
            trọng ở đây vì chunk cắt từ Bộ luật dài ngắn rất chênh nhau.

    TF-IDF — sklearn.TfidfVectorizer + cosine similarity
        score = cosine(tfidf(q), tfidf(d)), tf tuyến tính, chuẩn hoá L2.
        Chuẩn hoá L2 chỉ chia cho độ dài vector, không mô hình hoá "document dài
        thì đương nhiên chứa nhiều từ hơn" như tham số b của BM25.

Vì sao lexical search lại cần thiết bên cạnh semantic search (Task 5):
    Embedding rất kém với mã định danh. "Điều 25" và "Điều 52" nằm gần như cùng
    một chỗ trong không gian vector, còn "Nghị định 145/2020/NĐ-CP" thì bị tách
    thành các mảnh vô nghĩa. Nhưng đây lại chính là thứ người dùng tra cứu nhiều
    nhất trong hỏi đáp pháp luật. BM25 khớp chuỗi chính xác nên xử lý tốt đúng
    phần mà dense retrieval yếu nhất — đó là lý do hybrid search (Task 9) hơn hẳn
    dùng riêng một trong hai.

Cài đặt:
    pip install rank-bm25 scikit-learn

Chạy thử:
    python -m src.task6_lexical_search
"""

import re
import unicodedata
from pathlib import Path

import numpy as np

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Lấy tham số chunking từ Task 4 để corpus của BM25 khớp từng chunk với corpus
# của semantic search. Đây KHÔNG phải chi tiết vụn vặt: Task 7 gộp hai bảng xếp
# hạng bằng RRF với khoá là nội dung chunk. Nếu BM25 index nguyên file còn
# semantic index chunk 800 ký tự thì không khoá nào trùng nhau, RRF không cộng
# dồn được điểm cho bất kỳ tài liệu nào, và hybrid search thoái hoá thành phép
# nối hai danh sách — vẫn chạy, vẫn qua test, nhưng mất sạch tác dụng.
try:
    from src.task4_chunking_indexing import (
        CHROMA_DIR,
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        COLLECTION_NAME,
    )
except ImportError:  # Task 4 chưa xong -> dùng tham số theo yêu cầu bài lab
    CHUNK_SIZE, CHUNK_OVERLAP = 800, 100
    COLLECTION_NAME = "labor_law_docs"
    CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Tham số BM25 chuẩn (Robertson & Zaragoza 2009)
BM25_K1 = 1.5
BM25_B = 0.75


# =============================================================================
# Tokenizer cho tiếng Việt + văn bản pháp luật
# =============================================================================

# Giữ nguyên cụm số hiệu văn bản: "145/2020/nđ-cp", "45/2019/qh14"
_TOKEN_RE = re.compile(r"[0-9a-zà-ỹ]+(?:[/-][0-9a-zà-ỹ]+)*", re.IGNORECASE)


def normalize(text: str) -> str:
    """
    Chuẩn hoá Unicode NFC + lowercase.

    NFC là bắt buộc, không phải cho đẹp: tiếng Việt có hai cách mã hoá cùng một
    chữ — "ệ" có thể là 1 code point (U+1EC7) hoặc "e" + 2 dấu tổ hợp. Hai chuỗi
    đó hiển thị giống hệt nhau nhưng KHÁC nhau với ==. Corpus lấy từ web và câu
    hỏi người dùng gõ từ bàn phím rất hay lệch nhau ở điểm này, và khi lệch thì
    BM25 trượt hoàn toàn mà không báo lỗi gì cả.
    """
    return unicodedata.normalize("NFC", text).lower()


def tokenize(text: str) -> list[str]:
    """
    Tách token: unigram + bigram.

    Vì sao thêm bigram: tiếng Việt ghép nghĩa bằng nhiều âm tiết rời. Cắt theo
    khoảng trắng thì "thử việc" thành "thử" + "việc", mà "việc" là từ xuất hiện
    khắp nơi trong văn bản lao động nên gần như không phân biệt được gì. Bigram
    "thử_việc" khôi phục lại đơn vị nghĩa thật sự.

    Đặc biệt hữu ích với dẫn chiếu điều luật: "điều_25", "khoản_2" trở thành một
    token duy nhất, nên câu hỏi "Điều 25 quy định gì" bắn trúng đúng điều 25 thay
    vì mọi chunk có chứa số 25.

    Không loại stopword: IDF của BM25 đã tự hạ trọng số cho từ phổ biến, mà bỏ
    stopword thủ công lại phá vỡ tính liền kề khi tạo bigram.
    """
    unigrams = _TOKEN_RE.findall(normalize(text))
    bigrams = [f"{a}_{b}" for a, b in zip(unigrams, unigrams[1:])]
    return unigrams + bigrams


# =============================================================================
# Nạp corpus
# =============================================================================

def _chunk_text(text: str, source: str, doc_type: str) -> list[dict]:
    """Cắt 1 document thành chunk, dùng đúng tham số của Task 4."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        pieces = splitter.split_text(text)
    except ImportError:
        # Cắt thủ công có overlap khi chưa cài langchain-text-splitters
        pieces, start = [], 0
        while start < len(text):
            pieces.append(text[start:start + CHUNK_SIZE])
            start += CHUNK_SIZE - CHUNK_OVERLAP

    return [
        {
            "content": piece,
            "metadata": {"source": source, "type": doc_type, "chunk_index": i},
        }
        for i, piece in enumerate(pieces)
        if piece.strip()
    ]


def load_corpus_from_chroma() -> list[dict]:
    """
    Nạp chunk trực tiếp từ ChromaDB của Task 4.

    Đây là đường ưu tiên vì nó bảo đảm BM25 và semantic search nhìn thấy đúng
    cùng một tập chunk — điều kiện để RRF ở Task 7 gộp được điểm.
    """
    try:
        import chromadb
    except ImportError:
        return []

    if not Path(CHROMA_DIR).exists():
        return []

    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            collection = client.get_collection(COLLECTION_NAME)
        except Exception:
            # Role 2 có thể đặt tên collection khác -> lấy collection đầu tiên
            collections = client.list_collections()
            if not collections:
                return []
            name = collections[0] if isinstance(collections[0], str) else collections[0].name
            collection = client.get_collection(name)

        data = collection.get(include=["documents", "metadatas"])
    except Exception:
        return []

    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or [{}] * len(documents)

    return [
        {"content": doc, "metadata": meta or {}}
        for doc, meta in zip(documents, metadatas)
        if doc and doc.strip()
    ]


def load_corpus_from_markdown() -> list[dict]:
    """
    Nạp và cắt chunk từ data/standardized/ khi chưa có ChromaDB.

    Gọi thẳng hàm của Task 4 chứ không tự cắt lại: chỉ cần lệch một tham số
    (chunk_size, separator, ngưỡng bỏ chunk ngắn) là hai bên ra hai tập chunk
    khác nhau, và RRF ở Task 7 lập tức mất khả năng gộp điểm. Dùng chung một hàm
    thì sai lệch đó không thể xảy ra.
    """
    try:
        from src.task4_chunking_indexing import chunk_documents, load_documents
        return chunk_documents(load_documents())
    except ImportError:
        pass

    # Task 4 chưa sẵn sàng -> tự cắt bằng tham số tương đương
    if not STANDARDIZED_DIR.exists():
        return []

    corpus = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        if not text.strip():
            continue
        doc_type = "legal" if "legal" in md_file.parts else "news"
        corpus.extend(_chunk_text(text, md_file.name, doc_type))

    return corpus


def load_corpus() -> list[dict]:
    """Nạp corpus, ưu tiên ChromaDB rồi mới đến markdown."""
    corpus = load_corpus_from_chroma()
    if corpus:
        return corpus
    return load_corpus_from_markdown()


# =============================================================================
# Xây index
# =============================================================================

def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}

    Returns:
        BM25Okapi đã sẵn sàng truy vấn.
    """
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus, k1=BM25_K1, b=BM25_B)


def build_tfidf_index(corpus: list[dict]):
    """Xây dựng TF-IDF index (phương pháp thứ 2, dùng để so sánh trong demo)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    # analyzer=tokenize để TF-IDF dùng CHUNG bộ tokenizer với BM25 — có vậy thì
    # so sánh hai phương pháp mới công bằng, khác biệt đến từ công thức tính
    # điểm chứ không phải từ cách tách từ.
    vectorizer = TfidfVectorizer(analyzer=tokenize)
    matrix = vectorizer.fit_transform(doc["content"] for doc in corpus)
    return vectorizer, matrix


# Dựng index một lần cho mỗi tiến trình. Không cache ra file: corpus chỉ vài
# nghìn chunk nên build hết dưới 1 giây, trong khi cache trên đĩa lại đẻ ra lỗi
# cache cũ mỗi lần Role 2 reindex — đắt hơn nhiều so với thứ tiết kiệm được.
_CORPUS: list[dict] | None = None
_BM25 = None
_TFIDF = None


def _get_corpus() -> list[dict]:
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = load_corpus()
    return _CORPUS


def _get_bm25():
    global _BM25
    if _BM25 is None:
        corpus = _get_corpus()
        _BM25 = build_bm25_index(corpus) if corpus else None
    return _BM25


def _get_tfidf():
    global _TFIDF
    if _TFIDF is None:
        corpus = _get_corpus()
        _TFIDF = build_tfidf_index(corpus) if corpus else None
    return _TFIDF


def reset_index() -> None:
    """Xoá index đang giữ trong bộ nhớ (gọi sau khi corpus thay đổi)."""
    global _CORPUS, _BM25, _TFIDF
    _CORPUS = _BM25 = _TFIDF = None


# =============================================================================
# Tìm kiếm
# =============================================================================

def _top_results(scores, corpus: list[dict], top_k: int, source: str) -> list[dict]:
    """Lấy top_k kết quả có điểm > 0, sắp giảm dần."""
    if len(scores) == 0:
        return []

    # argpartition rẻ hơn sort toàn mảng khi chỉ cần top_k
    k = min(top_k, len(scores))
    candidate_idx = np.argpartition(scores, -k)[-k:]
    candidate_idx = candidate_idx[np.argsort(scores[candidate_idx])[::-1]]

    results = []
    for idx in candidate_idx:
        score = float(scores[idx])
        # Điểm 0 = không có token nào của câu hỏi xuất hiện trong chunk. Trả về
        # thì chỉ làm nhiễu Task 7 và tốn chỗ trong context của Task 10.
        if score <= 0:
            continue
        results.append({
            "content": corpus[idx]["content"],
            "score": round(score, 4),
            "metadata": corpus[idx].get("metadata", {}),
            "retriever": source,
        })

    return results


def lexical_search(query: str, top_k: int = 10, method: str = "bm25") -> list[dict]:
    """
    Tìm kiếm từ khoá (sparse retrieval).

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa
        method: "bm25" (mặc định) | "tfidf"

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'retriever': str
        }
        Sorted by score descending. Trả list rỗng nếu không có token nào khớp.
    """
    corpus = _get_corpus()
    if not corpus:
        return []

    if method == "bm25":
        bm25 = _get_bm25()
        if bm25 is None:
            return []
        scores = np.asarray(bm25.get_scores(tokenize(query)))

    elif method == "tfidf":
        tfidf = _get_tfidf()
        if tfidf is None:
            return []
        vectorizer, matrix = tfidf
        query_vec = vectorizer.transform([query])
        # matrix đã chuẩn hoá L2 nên tích vô hướng chính là cosine similarity
        scores = (matrix @ query_vec.T).toarray().ravel()

    else:
        raise ValueError(f"method phải là 'bm25' hoặc 'tfidf', nhận được: {method!r}")

    return _top_results(scores, corpus, top_k, f"lexical_{method}")


if __name__ == "__main__":
    corpus = _get_corpus()
    print("=" * 66)
    print("Task 6: Lexical Search (BM25 + TF-IDF)")
    print("=" * 66)
    print(f"Corpus: {len(corpus)} chunk (chunk_size={CHUNK_SIZE}, "
          f"overlap={CHUNK_OVERLAP})")

    if not corpus:
        print("\n⚠ Corpus rỗng — chạy Task 1-3 trước để sinh data/standardized/")
        raise SystemExit(1)

    demo_queries = [
        "thời gian thử việc tối đa cho lập trình viên là bao lâu",
        "lương thử việc tối thiểu bằng bao nhiêu phần trăm",
        "Điều 25",
        "sa thải không báo trước 30 ngày có đúng luật không",
    ]

    for query in demo_queries:
        print(f"\n{'─' * 66}")
        print(f"Truy vấn: {query!r}")
        for method in ("bm25", "tfidf"):
            results = lexical_search(query, top_k=3, method=method)
            print(f"\n  [{method.upper()}] {len(results)} kết quả")
            for rank, r in enumerate(results, 1):
                snippet = " ".join(r["content"].split())[:88]
                print(f"    {rank}. [{r['score']:.4f}] "
                      f"({r['metadata'].get('source', '?')})")
                print(f"       {snippet}...")
