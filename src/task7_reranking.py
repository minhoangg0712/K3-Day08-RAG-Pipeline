"""
Task 7 — Reranking Module.

Ba phương pháp, mặc định dùng RRF vì không cần API key và gộp được kết quả từ
semantic search (Task 5) với BM25 (Task 6):

    RRF (Reciprocal Rank Fusion) — Cormack et al. 2009
        RRF(d) = Σ_r 1 / (k + rank_r(d)),  k = 60
        Chỉ dùng THỨ HẠNG, không dùng điểm gốc. Đó vừa là ưu điểm vừa là bẫy:
        điểm cosine (thang [0,1]) và điểm BM25 (thang không chặn trên, ở corpus
        này lên tới ~50) không thể cộng trực tiếp — chuẩn hoá kiểu min-max thì
        lại phụ thuộc vào chính tập kết quả trả về, mỗi truy vấn một kiểu. RRF
        né toàn bộ vấn đề đó bằng cách chỉ so thứ hạng.

    Cross-encoder — Jina Reranker v2 multilingual (cần JINA_API_KEY)
    MMR (Maximal Marginal Relevance) — giảm trùng lặp, tăng đa dạng

⚠ BẪY QUAN TRỌNG NHẤT CỦA BÀI LAB (Task 9 phụ thuộc vào chỗ này):
    Điểm RRF sau khi fuse KHÔNG phản ánh độ liên quan. Với k=60, tài liệu hạng 1
    luôn có điểm ≈ 1/(60+1) = 0.0164 — bất kể nội dung có dính dáng gì tới câu
    hỏi hay không. Truy vấn "cách nấu phở" cũng cho ra top-1 đúng 0.0164 y hệt
    truy vấn "thời gian thử việc".

    Hệ quả: nếu Task 9 so score_threshold (0.48) với điểm RRF thì điều kiện
    0.0164 < 0.48 luôn đúng → fallback PageIndex kích hoạt ở MỌI truy vấn. Còn
    nếu so với ngưỡng nhỏ hơn 0.0164 thì fallback KHÔNG BAO GIỜ chạy. Cả hai
    trường hợp đều là hỏng logic, mà test vẫn xanh.

    Vì vậy các hàm ở đây luôn giữ lại điểm gốc của từng retriever trong
    'component_scores' để Task 9 lấy đúng điểm cosine của semantic search mà so
    ngưỡng, tách hẳn khỏi điểm dùng để sắp xếp.
"""

import os
from typing import Iterable

import numpy as np
from dotenv import load_dotenv

load_dotenv()

RRF_K = 60  # hằng số làm mượt, theo paper gốc Cormack et al. 2009
JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_MODEL = "jina-reranker-v2-base-multilingual"


# =============================================================================
# RRF — Reciprocal Rank Fusion
# =============================================================================

def rerank_rrf(
    ranked_lists: Iterable[list[dict]], top_k: int = 5, k: int = RRF_K
) -> list[dict]:
    """
    Gộp nhiều bảng xếp hạng thành một.

        RRF(d) = Σ_r 1 / (k + rank_r(d))

    Tài liệu được nhiều ranker cùng bình chọn sẽ cộng dồn điểm, nên nó nổi lên
    trên tài liệu chỉ được một ranker xếp hạng cao. Đây chính là cái mà hybrid
    search cần: đoạn văn vừa khớp ngữ nghĩa vừa chứa đúng số hiệu điều luật.

    Vì sao k = 60: k làm phẳng chênh lệch giữa các thứ hạng đầu. Với k nhỏ
    (ví dụ 1), hạng 1 được 0.5 còn hạng 2 chỉ 0.33 — một ranker duy nhất tự
    quyết định kết quả. Với k = 60, hạng 1 và hạng 2 chênh nhau chưa tới 2%, nên
    sự đồng thuận giữa nhiều ranker mới là yếu tố quyết định.

    Args:
        ranked_lists: Danh sách các bảng xếp hạng (mỗi bảng từ 1 ranker),
                      đã sắp giảm dần theo độ liên quan.
        top_k: Số kết quả cuối cùng
        k: Hằng số làm mượt

    Returns:
        List top_k, sắp giảm dần theo điểm RRF. Mỗi item giữ thêm:
            'rrf_score'       — điểm fuse (dùng để sắp xếp)
            'component_scores'— điểm GỐC của từng retriever (dùng cho Task 9)
            'ranks'           — thứ hạng ở từng retriever (dùng để debug/demo)
    """
    fused: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        if not ranked_list:
            continue

        for rank, item in enumerate(ranked_list, start=1):
            key = item.get("content", "")
            if not key:
                continue

            if key not in fused:
                fused[key] = {
                    **item,
                    "rrf_score": 0.0,
                    "component_scores": {},
                    "ranks": {},
                }

            entry = fused[key]
            entry["rrf_score"] += 1.0 / (k + rank)

            # Giữ nguyên điểm gốc để Task 9 còn so ngưỡng cosine cho đúng
            retriever = item.get("retriever", "unknown")
            entry["component_scores"][retriever] = item.get("score")
            entry["ranks"][retriever] = rank

    results = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)

    for item in results:
        # 'score' là điểm dùng để SẮP XẾP. Điểm gốc vẫn nằm nguyên trong
        # component_scores — đừng dùng 'score' ở đây để quyết định fallback.
        item["score"] = round(item["rrf_score"], 6)
        item["retriever"] = "rrf"

    return results[:top_k]


# =============================================================================
# Cross-encoder (Jina Reranker v2)
# =============================================================================

def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank bằng cross-encoder.

    Khác biệt so với bi-encoder ở Task 5: bi-encoder embed câu hỏi và tài liệu
    RIÊNG BIỆT rồi mới so vector, nên hai bên không bao giờ "nhìn thấy" nhau.
    Cross-encoder đưa cả cặp (câu hỏi, tài liệu) vào cùng một lượt forward, cho
    phép attention chạy chéo giữa từng từ của câu hỏi và từng từ của tài liệu —
    chính xác hơn nhiều, nhưng phải chạy lại cho mỗi cặp nên chỉ dùng được ở bước
    rerank trên vài chục ứng viên, không dùng để quét cả corpus.

    Returns:
        List top_k đã chấm điểm lại. Nếu không có API key hoặc gọi lỗi thì trả
        về candidates cắt top_k (giữ nguyên thứ tự) để pipeline không gãy.
    """
    if not candidates:
        return []

    if not JINA_API_KEY:
        print("  ⚠ Chưa có JINA_API_KEY — bỏ qua cross-encoder, giữ thứ tự cũ")
        return candidates[:top_k]

    try:
        import requests

        response = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers={"Authorization": f"Bearer {JINA_API_KEY}"},
            json={
                "model": JINA_MODEL,
                "query": query,
                "documents": [c["content"] for c in candidates],
                "top_n": top_k,
            },
            timeout=30,
        )
        response.raise_for_status()
        reranked = response.json()["results"]
    except Exception as e:
        print(f"  ⚠ Jina rerank lỗi ({type(e).__name__}), giữ thứ tự cũ: {e}")
        return candidates[:top_k]

    output = []
    for r in reranked:
        original = candidates[r["index"]]
        output.append({
            **original,
            "score": round(float(r["relevance_score"]), 6),
            "component_scores": {
                **original.get("component_scores", {}),
                original.get("retriever", "unknown"): original.get("score"),
            },
            "retriever": "cross_encoder",
        })

    return output[:top_k]


# =============================================================================
# MMR — Maximal Marginal Relevance
# =============================================================================

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity giữa 2 vector."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Chọn kết quả vừa liên quan vừa đa dạng.

        MMR = λ·sim(query, doc) − (1−λ)·max sim(doc, đã_chọn)

    Vì sao cần: chunk có overlap 100 ký tự nên hai chunk kề nhau chia sẻ một
    đoạn text. Top-5 thuần theo độ liên quan rất dễ là 5 chunk liên tiếp của cùng
    một điều luật — nhìn thì điểm cao nhưng thực chất chỉ nói một ý, và Task 10
    mất sạch chỗ cho các điều luật liên quan khác. MMR trừ điểm những ứng viên
    quá giống thứ đã chọn nên buộc kết quả phải trải rộng.

    λ = 0.7: nghiêng về độ liên quan, chỉ dùng 30% trọng số cho đa dạng. Trong
    hỏi đáp pháp luật, trả lời đúng điều luật quan trọng hơn là trả lời đa dạng.

    Args:
        query_embedding: Vector của câu hỏi
        candidates: List of {'content', 'score', 'embedding', 'metadata'}
        top_k: Số kết quả
        lambda_param: 1.0 = chỉ quan tâm liên quan, 0.0 = chỉ quan tâm đa dạng
    """
    usable = [c for c in candidates if c.get("embedding") is not None]
    if not usable:
        print("  ⚠ Candidates không có 'embedding' — MMR cần vector, giữ thứ tự cũ")
        return candidates[:top_k]

    query_vec = np.asarray(query_embedding, dtype=np.float32)
    vectors = [np.asarray(c["embedding"], dtype=np.float32) for c in usable]
    relevance = [_cosine(query_vec, v) for v in vectors]

    selected: list[int] = []
    remaining = list(range(len(usable)))

    while remaining and len(selected) < top_k:
        best_idx, best_score = None, float("-inf")

        for idx in remaining:
            max_sim_selected = max(
                (_cosine(vectors[idx], vectors[s]) for s in selected), default=0.0
            )
            mmr = lambda_param * relevance[idx] - (1 - lambda_param) * max_sim_selected

            if mmr > best_score:
                best_score, best_idx = mmr, idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    output = []
    for rank, idx in enumerate(selected):
        item = usable[idx]
        output.append({
            **item,
            "score": round(float(relevance[idx]), 6),
            "component_scores": {
                **item.get("component_scores", {}),
                item.get("retriever", "unknown"): item.get("score"),
            },
            "retriever": "mmr",
        })

    return output


# =============================================================================
# Fallback rerank khi chỉ có 1 danh sách và không có API key
# =============================================================================

def rerank_lexical_overlap(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Chấm lại theo tỉ lệ token của câu hỏi xuất hiện trong tài liệu.

    Dùng khi chỉ có một danh sách ứng viên (RRF cần ít nhất hai) và không có
    JINA_API_KEY. Không mạnh bằng cross-encoder nhưng chạy offline, không tốn
    tiền, và vẫn là rerank thật — nó đo mức phủ của câu hỏi trên tài liệu chứ
    không phải giữ nguyên thứ tự cũ.

    Dùng chung tokenizer với Task 6 nên các cụm ghép ("thử_việc") và số hiệu
    điều luật ("điều_25") được tính là một đơn vị khớp.
    """
    if not candidates:
        return []

    try:
        from src.task6_lexical_search import tokenize
    except ImportError:
        return candidates[:top_k]

    query_tokens = set(tokenize(query))
    if not query_tokens:
        return candidates[:top_k]

    scored = []
    for item in candidates:
        doc_tokens = set(tokenize(item.get("content", "")))
        overlap = len(query_tokens & doc_tokens) / len(query_tokens)
        scored.append({
            **item,
            "score": round(overlap, 6),
            "component_scores": {
                **item.get("component_scores", {}),
                item.get("retriever", "unknown"): item.get("score"),
            },
            "retriever": "lexical_overlap",
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# =============================================================================
# Giao diện thống nhất
# =============================================================================

def _is_list_of_lists(candidates) -> bool:
    """Phân biệt [[...], [...]] (nhiều bảng xếp hạng) với [{...}] (một bảng)."""
    return bool(candidates) and isinstance(candidates[0], list)


def rerank(
    query: str,
    candidates: list,
    top_k: int = 5,
    method: str = "auto",
) -> list[dict]:
    """
    Giao diện rerank thống nhất, nhận cả 2 dạng đầu vào.

    Args:
        query: Câu truy vấn
        candidates: Một trong hai dạng
            - list[list[dict]]: nhiều bảng xếp hạng → gộp bằng RRF
            - list[dict]:       một bảng ứng viên  → chấm lại từng cái
        top_k: Số kết quả
        method: "auto" | "rrf" | "cross_encoder" | "lexical_overlap"
            "auto" tự chọn: nhiều bảng → RRF; một bảng → cross-encoder nếu có
            API key, không thì lexical_overlap.

    Returns:
        List tối đa top_k kết quả đã rerank, luôn có trường 'score'.
    """
    if not candidates:
        return []

    multi = _is_list_of_lists(candidates)

    if method == "auto":
        if multi:
            method = "rrf"
        else:
            method = "cross_encoder" if JINA_API_KEY else "lexical_overlap"

    if method == "rrf":
        # Một bảng đơn lẻ vẫn hợp lệ với RRF (thành phép đổi thang theo thứ hạng)
        ranked_lists = candidates if multi else [candidates]
        return rerank_rrf(ranked_lists, top_k=top_k)

    # Các phương pháp còn lại chấm điểm trên từng tài liệu -> làm phẳng đầu vào
    flat = [item for sublist in candidates for item in sublist] if multi else candidates

    if method == "cross_encoder":
        return rerank_cross_encoder(query, flat, top_k=top_k)

    if method == "lexical_overlap":
        return rerank_lexical_overlap(query, flat, top_k=top_k)

    if method == "mmr":
        raise ValueError("MMR cần query_embedding — gọi trực tiếp rerank_mmr()")

    raise ValueError(f"Phương pháp rerank không hợp lệ: {method!r}")


if __name__ == "__main__":
    print("=" * 66)
    print("Task 7: Reranking — minh hoạ RRF gộp 2 ranker")
    print("=" * 66)

    semantic = [
        {"content": "Điều 26. Tiền lương thử việc ít nhất bằng 85% mức lương",
         "score": 0.71, "metadata": {"source": "blld.md"}, "retriever": "semantic"},
        {"content": "Điều 24. Thử việc: hai bên có thể thỏa thuận nội dung thử việc",
         "score": 0.66, "metadata": {"source": "blld.md"}, "retriever": "semantic"},
        {"content": "Điều 113. Nghỉ hằng năm của người lao động",
         "score": 0.41, "metadata": {"source": "blld.md"}, "retriever": "semantic"},
    ]
    lexical = [
        {"content": "Điều 25. Thời gian thử việc không quá 60 ngày",
         "score": 31.2, "metadata": {"source": "blld.md"}, "retriever": "lexical_bm25"},
        {"content": "Điều 26. Tiền lương thử việc ít nhất bằng 85% mức lương",
         "score": 29.8, "metadata": {"source": "blld.md"}, "retriever": "lexical_bm25"},
        {"content": "Điều 24. Thử việc: hai bên có thể thỏa thuận nội dung thử việc",
         "score": 12.4, "metadata": {"source": "blld.md"}, "retriever": "lexical_bm25"},
    ]

    print("\nSemantic (cosine, thang [0,1]) và BM25 (thang không chặn trên ~50)")
    print("không thể cộng trực tiếp — RRF chỉ dùng thứ hạng nên né được vấn đề.\n")

    fused = rerank_rrf([semantic, lexical], top_k=4)
    for rank, r in enumerate(fused, 1):
        print(f"{rank}. RRF={r['score']:.6f}  {r['content'][:58]}")
        print(f"   thứ hạng gốc: {r['ranks']}")
        print(f"   điểm gốc    : {r['component_scores']}")

    print(f"\n→ Hai hạng đầu ({fused[0]['score']:.6f}, {fused[1]['score']:.6f}) cao hơn")
    print("  vì được CẢ HAI ranker bình chọn — đúng tác dụng mong muốn của RRF.")
    print(f"\n→ Bẫy: tài liệu chỉ được 1 ranker xếp hạng 1 luôn được đúng")
    print(f"  1/(60+1) = {1/61:.6f} — xem dòng 3 ở trên. Con số đó KHÔNG đổi")
    print("  kể cả khi truy vấn hoàn toàn lạc đề, vì RRF chỉ nhìn thứ hạng.")
    print("  Nên Task 9 phải so ngưỡng 0.48 với component_scores['semantic'],")
    print("  tuyệt đối không so với 'score' đã fuse.")
