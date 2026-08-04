"""
Task 5 — Semantic Search Module (Dense Retrieval).

Tìm theo ngữ nghĩa trên ChromaDB đã index ở Task 4, kèm HyDE (tiêu chí bonus +5đ).

Điểm trả về là COSINE SIMILARITY thật trong khoảng [0, 1], không phải khoảng cách
và cũng không phải điểm đã fuse. Task 9 so ngưỡng fallback (0.48) với chính điểm
này — xem ghi chú trong Task 7 về việc vì sao không được so ngưỡng với điểm RRF.

Chạy thử:
    python -m src.task5_semantic_search
"""

import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

try:
    from src.task4_chunking_indexing import (
        CHROMA_DIR,
        COLLECTION_NAME,
        EMBEDDING_MODEL,
    )
except ImportError:  # Task 4 chưa xong
    CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
    COLLECTION_NAME = "labor_law_docs"
    EMBEDDING_MODEL = "BAAI/bge-m3"

# Model sinh câu trả lời giả định cho HyDE. Dùng bản :free của OpenRouter vì
# HyDE gọi LLM ở MỌI truy vấn — chi phí cộng dồn rất nhanh nếu dùng model trả phí.
HYDE_MODEL = os.getenv("HYDE_MODEL", "google/gemma-4-26b-a4b-it:free")


# =============================================================================
# Model & collection (khởi tạo trễ, dùng lại trong suốt tiến trình)
# =============================================================================

_MODEL = None
_COLLECTION = None


def get_embedding_model():
    """
    Nạp embedding model một lần duy nhất.

    Model phải TRÙNG với model đã dùng ở Task 4. Embed câu hỏi bằng một model
    khác với model đã embed corpus thì hai vector nằm ở hai không gian không liên
    quan gì nhau — kết quả trả về trông vẫn bình thường (vẫn có điểm, vẫn xếp
    hạng) nhưng thực chất là ngẫu nhiên. Đây là lỗi im lặng, rất khó phát hiện.
    """
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _MODEL


def get_collection():
    """Lấy collection ChromaDB đã tạo ở Task 4."""
    global _COLLECTION
    if _COLLECTION is None:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            _COLLECTION = client.get_collection(COLLECTION_NAME)
        except Exception:
            # Tên collection có thể khác nếu Role 2 đổi -> lấy cái đang có
            collections = client.list_collections()
            if not collections:
                raise RuntimeError(
                    f"Chưa có collection nào trong {CHROMA_DIR}. "
                    f"Chạy `python -m src.task4_chunking_indexing` trước."
                )
            first = collections[0]
            name = first if isinstance(first, str) else first.name
            _COLLECTION = client.get_collection(name)
    return _COLLECTION


def reset_cache() -> None:
    """Xoá model/collection đang giữ trong bộ nhớ."""
    global _MODEL, _COLLECTION
    _MODEL = _COLLECTION = None


def embed_query(text: str) -> np.ndarray:
    """Embed một câu và chuẩn hoá L2 để tích vô hướng chính là cosine."""
    model = get_embedding_model()
    vector = model.encode(text, normalize_embeddings=True)
    return np.asarray(vector, dtype=np.float32)


# =============================================================================
# HyDE — Hypothetical Document Embeddings
# =============================================================================

HYDE_PROMPT = """Bạn là chuyên gia pháp luật lao động Việt Nam.

Hãy viết một đoạn văn NGẮN (3-4 câu) trả lời câu hỏi dưới đây theo đúng văn phong
của văn bản quy phạm pháp luật: dùng thuật ngữ pháp lý chuẩn ("người lao động",
"người sử dụng lao động", "hợp đồng lao động"), nêu số điều luật nếu bạn biết.

Chỉ viết đoạn văn, không giải thích thêm. Không cần chính xác tuyệt đối.

Câu hỏi: {query}"""


def generate_hypothetical_document(query: str) -> str | None:
    """
    Sinh câu trả lời giả định cho HyDE.

    Ý tưởng: câu hỏi và văn bản luật viết bằng hai thứ ngôn ngữ khác nhau. Người
    dùng hỏi "bị đuổi việc qua Zalo có sai không", còn luật viết "đơn phương chấm
    dứt hợp đồng lao động trái pháp luật". Khoảng cách từ vựng đó khiến embedding
    của câu hỏi không nằm gần embedding của điều luật cần tìm.

    HyDE bắc cầu bằng cách cho LLM viết trước một đoạn TRẢ LỜI giả định theo văn
    phong pháp lý, rồi đem chính đoạn đó đi tìm kiếm. Đoạn giả định dù có sai chi
    tiết vẫn dùng đúng lớp từ vựng của văn bản luật, nên nó nằm gần vùng cần tìm
    hơn hẳn câu hỏi gốc.

    Returns:
        Đoạn văn giả định, hoặc None nếu không gọi được LLM.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        response = client.chat.completions.create(
            model=HYDE_MODEL,
            messages=[{"role": "user", "content": HYDE_PROMPT.format(query=query)}],
            # temperature thấp: cần đoạn văn đúng văn phong pháp lý, không cần
            # sáng tạo. Nhiệt độ cao chỉ làm tăng nguy cơ bịa ra điều luật lạ và
            # kéo truy vấn đi chệch hướng.
            temperature=0.3,
            max_tokens=250,
        )
        text = (response.choices[0].message.content or "").strip()
        return text or None
    except Exception as e:
        print(f"  ⚠ HyDE thất bại ({type(e).__name__}), dùng câu hỏi gốc: {e}")
        return None


def build_query_vector(query: str, use_hyde: bool = False) -> np.ndarray:
    """
    Dựng vector truy vấn, có hoặc không có HyDE.

    Khi bật HyDE thì lấy TRUNG BÌNH của vector câu hỏi gốc và vector đoạn giả
    định, chứ không thay thế hẳn. Lý do: nếu LLM bịa nhầm sang một chế định khác
    (hỏi về thử việc mà nó viết về hợp đồng học nghề) thì HyDE thuần sẽ kéo toàn
    bộ truy vấn đi lạc. Giữ lại câu hỏi gốc làm mỏ neo giúp sai lệch đó không
    thành thảm hoạ, mà vẫn hưởng phần lớn lợi ích về từ vựng.
    """
    query_vec = embed_query(query)

    if not use_hyde:
        return query_vec

    hypothetical = generate_hypothetical_document(query)
    if not hypothetical:
        return query_vec

    hyde_vec = embed_query(hypothetical)
    combined = (query_vec + hyde_vec) / 2.0

    # Chuẩn hoá lại: trung bình của 2 vector đơn vị không còn là vector đơn vị
    norm = np.linalg.norm(combined)
    return combined / norm if norm > 0 else query_vec


# =============================================================================
# Tìm kiếm
# =============================================================================

def semantic_search(query: str, top_k: int = 10, use_hyde: bool = False) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa bằng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa
        use_hyde: Bật HyDE (tốn thêm 1 lượt gọi LLM cho mỗi truy vấn)

    Returns:
        List of {
            'content': str,
            'score': float,       # cosine similarity trong [0, 1]
            'metadata': dict,
            'retriever': str
        }
        Sorted by score descending.
    """
    try:
        collection = get_collection()
    except Exception as e:
        print(f"  ⚠ Chưa truy cập được vector store: {e}")
        return []

    query_vec = build_query_vector(query, use_hyde=use_hyde)

    results = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    output = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        # Chroma với hnsw:space=cosine trả về cosine DISTANCE = 1 - cosine_sim,
        # nằm trong [0, 2]. Đổi ngược lại và kẹp về [0, 1] vì Task 9 so ngưỡng
        # 0.48 trên thang similarity.
        similarity = max(0.0, min(1.0, 1.0 - float(dist)))
        output.append({
            "content": doc,
            "score": round(similarity, 4),
            "metadata": meta or {},
            "retriever": "semantic",
        })

    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    print("=" * 66)
    print("Task 5: Semantic Search (Dense Retrieval + HyDE)")
    print("=" * 66)
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Chroma dir: {CHROMA_DIR}")

    demo_queries = [
        "thời gian thử việc tối đa cho vị trí lập trình viên là bao lâu",
        "công ty sa thải qua tin nhắn Zalo không báo trước có đúng luật không",
    ]

    for query in demo_queries:
        print(f"\n{'─' * 66}")
        print(f"Truy vấn: {query!r}")

        for use_hyde in (False, True):
            label = "HyDE" if use_hyde else "Thường"
            results = semantic_search(query, top_k=3, use_hyde=use_hyde)
            print(f"\n  [{label}] {len(results)} kết quả")
            for rank, r in enumerate(results, 1):
                snippet = " ".join(r["content"].split())[:88]
                print(f"    {rank}. [{r['score']:.4f}] "
                      f"({r['metadata'].get('source', '?')})")
                print(f"       {snippet}...")
