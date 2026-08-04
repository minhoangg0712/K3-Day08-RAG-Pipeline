"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement — khuyến nghị vì không cần API key

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.

Lưu ý quan trọng về RRF (sẽ dùng lại ở Task 9): điểm RRF fused CHỈ phụ thuộc thứ hạng,
không phải độ tương đồng thật. Top-1 sau khi fuse luôn xấp xỉ 1/(k+1) ≈ 0.0164 (k=60),
bất kể nội dung đó có thật sự liên quan đến câu hỏi hay không. Đừng dùng điểm RRF để
quyết định fallback ở Task 9 — xem ghi chú ở đó.
"""

def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    # TODO: Implement cross-encoder reranking
    #
    # Option A: Jina Reranker API
    # import requests
    # response = requests.post(
    #     "https://api.jina.ai/v1/rerank",
    #     headers={"Authorization": f"Bearer {JINA_API_KEY}"},
    #     json={
    #         "model": "jina-reranker-v2-base-multilingual",
    #         "query": query,
    #         "documents": [c["content"] for c in candidates],
    #         "top_n": top_k
    #     }
    # )
    # reranked = response.json()["results"]
    # return [
    #     {**candidates[r["index"]], "score": r["relevance_score"]}
    #     for r in reranked
    # ]
    #
    # Option B: Local model (Qwen3-Reranker)
    # from transformers import AutoModelForSequenceClassification, AutoTokenizer
    # ...
    raise NotImplementedError("Implement rerank_cross_encoder")


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    # TODO: Implement MMR
    #
    # selected = []
    # remaining = list(range(len(candidates)))
    #
    # for _ in range(min(top_k, len(candidates))):
    #     best_idx = None
    #     best_score = float('-inf')
    #
    #     for idx in remaining:
    #         # Relevance to query
    #         relevance = cosine_sim(query_embedding, candidates[idx]["embedding"])
    #
    #         # Max similarity to already selected
    #         max_sim_to_selected = 0
    #         for sel_idx in selected:
    #             sim = cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
    #             max_sim_to_selected = max(max_sim_to_selected, sim)
    #
    #         # MMR score
    #         mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
    #
    #         if mmr_score > best_score:
    #             best_score = mmr_score
    #             best_idx = idx
    #
    #     selected.append(best_idx)
    #     remaining.remove(best_idx)
    #
    # return [candidates[i] for i in selected]
    raise NotImplementedError("Implement rerank_mmr")


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    if top_k <= 0 or not ranked_lists:
        return []
    if k < 0:
        raise ValueError("k must be greater than or equal to 0")
    if not isinstance(ranked_lists, list):
        raise TypeError("ranked_lists must be a list of ranked result lists")

    rrf_scores: dict[tuple, float] = {}
    candidates_by_key: dict[tuple, dict] = {}
    best_ranks: dict[tuple, int] = {}
    first_seen: dict[tuple, int] = {}
    seen_counter = 0

    for list_index, ranked_list in enumerate(ranked_lists):
        if not isinstance(ranked_list, list):
            raise TypeError(f"ranked_lists[{list_index}] must be a list")

        # Một tài liệu chỉ được tính một lần trong cùng một ranker. Nếu ranker
        # vô tình trả trùng, lần xuất hiện đầu tiên (hạng tốt nhất) được giữ lại.
        seen_in_this_list: set[tuple] = set()

        for rank, item in enumerate(ranked_list, start=1):
            if not isinstance(item, dict):
                raise TypeError(
                    f"ranked_lists[{list_index}][{rank - 1}] must be a dict"
                )

            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Each candidate must contain non-empty 'content'")

            metadata = item.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}

            # Ưu tiên ID ổn định do Task 4 tạo. Nếu dữ liệu từ ranker khác
            # không có ID, nội dung chunk là khóa chung để fusion hai danh sách.
            chunk_id = item.get("id") or metadata.get("chunk_id")
            if chunk_id is not None:
                key = ("id", str(chunk_id))
            else:
                key = ("content", content.strip())

            if key in seen_in_this_list:
                continue
            seen_in_this_list.add(key)

            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            best_ranks[key] = min(best_ranks.get(key, rank), rank)

            if key not in candidates_by_key:
                candidate = item.copy()
                candidate["metadata"] = metadata.copy()
                candidates_by_key[key] = candidate
                first_seen[key] = seen_counter
                seen_counter += 1

    # Tie-break theo hạng tốt nhất rồi thứ tự xuất hiện để kết quả luôn ổn định.
    ordered_keys = sorted(
        rrf_scores,
        key=lambda key: (-rrf_scores[key], best_ranks[key], first_seen[key]),
    )

    results: list[dict] = []
    for key in ordered_keys[:top_k]:
        candidate = candidates_by_key[key].copy()
        candidate["metadata"] = candidates_by_key[key]["metadata"].copy()
        candidate["score"] = rrf_scores[key]
        results.append(candidate)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict] | list[list[dict]],
    top_k: int = 5,
    method: str = "rrf",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Một danh sách candidates đã xếp hạng, hoặc nhiều danh sách
            từ các ranker khi dùng RRF.
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    method = method.lower().strip()

    if method == "cross_encoder":
        if candidates and isinstance(candidates[0], list):
            raise TypeError("cross_encoder expects one flat candidate list")
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        # Cần query_embedding - embed query trước
        raise NotImplementedError("Call rerank_mmr with query_embedding")
    elif method == "rrf":
        if not candidates:
            return []

        # API chung trong bộ test truyền một danh sách phẳng. Coi nó như kết
        # quả của một ranker; khi tích hợp Task 5 + 6 có thể truyền [dense, sparse].
        if isinstance(candidates[0], list):
            if not all(isinstance(ranked_list, list) for ranked_list in candidates):
                raise TypeError("RRF candidates must all be ranked lists")
            return rerank_rrf(candidates, top_k=top_k)

        if not all(isinstance(item, dict) for item in candidates):
            raise TypeError("RRF candidates must be dictionaries")
        return rerank_rrf([candidates], top_k=top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    dense_results = [
        {"content": "Lương thử việc tối thiểu bằng 85%.", "score": 0.86, "metadata": {}},
        {"content": "Thử việc có thể kéo dài tối đa 60 ngày.", "score": 0.80, "metadata": {}},
    ]
    sparse_results = [
        {"content": "Thử việc có thể kéo dài tối đa 60 ngày.", "score": 7.2, "metadata": {}},
        {"content": "Lương thử việc tối thiểu bằng 85%.", "score": 5.8, "metadata": {}},
    ]

    results = rerank_rrf([dense_results, sparse_results], top_k=2)
    for result in results:
        print(f"[{result['score']:.6f}] {result['content']}")
