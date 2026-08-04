"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

    Query
      ├─→ Semantic Search (Task 5)  ──┐
      │                                ├─→ RRF Fusion (Task 7) → Results
      ├─→ Lexical Search  (Task 6)  ──┘
      │
      └─→ Nếu điểm cosine GỐC cao nhất < score_threshold
            └─→ Fallback: PageIndex Vectorless (Task 8)

Chạy:
    python -m src.task9_retrieval_pipeline
"""

from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search
from src.task7_reranking import rerank_rrf
from src.task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

# Ngưỡng kích hoạt fallback, đo trên COSINE SIMILARITY gốc của semantic search.
# 0.48 theo yêu cầu bài lab. Ngưỡng này phụ thuộc embedding model và corpus, nên
# sau khi Task 4 index xong hãy hiệu chỉnh lại bằng calibrate_threshold() ở cuối
# file: chạy vài câu chắc chắn liên quan và vài câu chắc chắn lạc đề, xem hai
# nhóm điểm tách nhau ở đâu rồi chọn ngưỡng nằm giữa.
SCORE_THRESHOLD = 0.48
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"

# Lấy dư ứng viên trước khi fuse. RRF cần nhìn đủ sâu vào mỗi bảng xếp hạng thì
# mới phát hiện được tài liệu nào được CẢ HAI retriever cùng bình chọn — nếu mỗi
# bên chỉ đưa lên đúng top_k thì phần giao gần như luôn rỗng, và RRF suy biến
# thành phép nối hai danh sách.
CANDIDATE_MULTIPLIER = 4


def _best_semantic_score(semantic_results: list[dict]) -> float:
    """Điểm cosine cao nhất của semantic search; 0.0 nếu không có kết quả."""
    return max((r.get("score", 0.0) for r in semantic_results), default=0.0)


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
    use_hyde: bool = False,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Args:
        query: Câu truy vấn
        top_k: Số kết quả cuối cùng
        score_threshold: Ngưỡng cosine GỐC (không phải điểm RRF)
        use_reranking: Có gộp bằng RRF hay chỉ nối kết quả
        use_hyde: Bật HyDE cho semantic search

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'hybrid' | 'pageindex'
        }
    """
    n_candidates = max(top_k * CANDIDATE_MULTIPLIER, 10)

    # --- Bước 1: chạy 2 retriever -----------------------------------------
    # Bọc try/except riêng từng nhánh: thiếu ChromaDB hay thiếu corpus chỉ được
    # phép làm mất MỘT nhánh, không được kéo cả pipeline sập theo.
    try:
        dense_results = semantic_search(query, top_k=n_candidates, use_hyde=use_hyde)
    except Exception as e:
        print(f"  ⚠ Semantic search lỗi ({type(e).__name__}): {e}")
        dense_results = []

    try:
        sparse_results = lexical_search(query, top_k=n_candidates)
    except Exception as e:
        print(f"  ⚠ Lexical search lỗi ({type(e).__name__}): {e}")
        sparse_results = []

    # --- Bước 2: gộp bằng RRF ---------------------------------------------
    if use_reranking:
        merged = rerank_rrf([dense_results, sparse_results], top_k=top_k)
    else:
        merged = (dense_results + sparse_results)[:top_k]

    for item in merged:
        item["source"] = "hybrid"

    # --- Bước 3: quyết định fallback --------------------------------------
    #
    # CHỖ DỄ SAI NHẤT CỦA CẢ BÀI LAB.
    #
    # So ngưỡng với điểm COSINE GỐC của semantic search, KHÔNG PHẢI điểm RRF đã
    # fuse. Điểm RRF chỉ phụ thuộc thứ hạng: tài liệu đứng đầu luôn được đúng
    # 1/(60+1) = 0.0164 bất kể nội dung có liên quan hay không. Nếu đem 0.0164
    # so với ngưỡng 0.48 thì điều kiện luôn đúng → fallback chạy ở MỌI truy vấn;
    # còn hạ ngưỡng xuống dưới 0.0164 cho "hợp thang điểm RRF" thì fallback
    # KHÔNG BAO GIỜ chạy, kể cả với câu hỏi hoàn toàn vô nghĩa.
    #
    # Cả hai trường hợp đều hỏng logic mà test vẫn xanh — nên lỗi này rất khó
    # phát hiện nếu không chủ động kiểm tra bằng một câu hỏi lạc đề.
    #
    # Cosine similarity nằm trong [0,1] và thực sự đo độ liên quan, nên nó mới
    # là đại lượng đúng để so ngưỡng.
    best_score = _best_semantic_score(dense_results)

    if merged and best_score >= score_threshold:
        return merged[:top_k]

    if merged:
        print(f"  ⚠ Cosine tốt nhất ({best_score:.3f}) < ngưỡng ({score_threshold}) "
              f"→ thử fallback PageIndex")

    fallback = pageindex_search(query, top_k=top_k)
    if fallback:
        return fallback[:top_k]

    # PageIndex chưa cấu hình được. Trả kết quả hybrid dù điểm thấp vẫn hơn trả
    # rỗng: Task 10 vẫn có quyền tự kết luận "không đủ căn cứ" từ nội dung nhận
    # được, còn trả rỗng thì nó không có gì để đánh giá cả.
    return merged[:top_k]


def retrieve_verbose(query: str, top_k: int = DEFAULT_TOP_K, **kwargs) -> dict:
    """Bản kèm thông tin chẩn đoán — dùng cho demo và cho app.py."""
    n_candidates = max(top_k * CANDIDATE_MULTIPLIER, 10)

    try:
        dense = semantic_search(query, top_k=n_candidates)
    except Exception:
        dense = []
    try:
        sparse = lexical_search(query, top_k=n_candidates)
    except Exception:
        sparse = []

    threshold = kwargs.get("score_threshold", SCORE_THRESHOLD)
    best = _best_semantic_score(dense)
    results = retrieve(query, top_k=top_k, **kwargs)

    return {
        "query": query,
        "results": results,
        "diagnostics": {
            "n_semantic": len(dense),
            "n_lexical": len(sparse),
            "best_cosine": round(best, 4),
            "threshold": threshold,
            "below_threshold": best < threshold,
            "used_pageindex": any(r.get("source") == "pageindex" for r in results),
        },
    }


def calibrate_threshold(
    relevant_queries: list[str], off_topic_queries: list[str]
) -> None:
    """
    Đo khoảng điểm cosine để chọn ngưỡng fallback cho đúng corpus của mình.

    Đừng copy nguyên ngưỡng của người khác: mỗi embedding model và mỗi corpus
    cho một dải điểm khác nhau. Cách làm: chạy vài câu chắc chắn liên quan và
    vài câu chắc chắn lạc đề, xem hai nhóm tách nhau ở đâu, chọn ngưỡng ở giữa.
    """
    def scores(queries):
        out = []
        for q in queries:
            try:
                out.append((q, _best_semantic_score(semantic_search(q, top_k=5))))
            except Exception as e:
                print(f"  ⚠ {q!r}: {type(e).__name__}: {e}")
        return out

    print("\n--- Câu hỏi LIÊN QUAN ---")
    rel = scores(relevant_queries)
    for q, s in rel:
        print(f"  {s:.4f}  {q[:58]}")

    print("\n--- Câu hỏi LẠC ĐỀ ---")
    off = scores(off_topic_queries)
    for q, s in off:
        print(f"  {s:.4f}  {q[:58]}")

    if rel and off:
        lo = min(s for _, s in rel)
        hi = max(s for _, s in off)
        print(f"\nThấp nhất của nhóm liên quan: {lo:.4f}")
        print(f"Cao nhất của nhóm lạc đề    : {hi:.4f}")
        if lo > hi:
            print(f"→ Hai nhóm tách bạch. Ngưỡng gợi ý: {(lo + hi) / 2:.4f}")
        else:
            print("→ Hai nhóm CHỒNG LẤN — không có ngưỡng nào tách sạch được. "
                  "Cân nhắc đổi embedding model hoặc bổ sung dữ liệu.")


if __name__ == "__main__":
    print("=" * 66)
    print("Task 9: Retrieval Pipeline (Hybrid + RRF + Fallback)")
    print("=" * 66)
    print(f"Ngưỡng fallback: cosine < {SCORE_THRESHOLD}")

    demo_queries = [
        "Thời gian thử việc tối đa cho vị trí lập trình viên là bao lâu?",
        "Công ty sa thải tôi qua tin nhắn Zalo mà không báo trước 30 ngày "
        "thì có đúng luật không?",
        "cách nấu phở bò gia truyền",  # lạc đề → phải kích hoạt fallback
    ]

    for query in demo_queries:
        print(f"\n{'─' * 66}")
        print(f"Truy vấn: {query!r}")

        out = retrieve_verbose(query, top_k=3)
        d = out["diagnostics"]
        print(f"  semantic={d['n_semantic']} lexical={d['n_lexical']} "
              f"| cosine tốt nhất={d['best_cosine']} | ngưỡng={d['threshold']}")
        print(f"  → dưới ngưỡng: {d['below_threshold']} "
              f"| dùng PageIndex: {d['used_pageindex']}")

        for rank, r in enumerate(out["results"], 1):
            snippet = " ".join(r["content"].split())[:76]
            print(f"    {rank}. [{r['source']}] {snippet}...")
