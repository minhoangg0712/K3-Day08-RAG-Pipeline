"""
Task 4 — Chunking & Indexing vào Vector Store.

Luồng: data/standardized/*.md → cắt chunk → embedding → ChromaDB (chroma_db/).

Chạy:
    python -m src.task4_chunking_indexing              # reindex sạch (mặc định)
    python -m src.task4_chunking_indexing --no-reset   # thêm vào index đang có

Lưu ý quan trọng: mặc định script XOÁ collection cũ trước khi index lại. Nếu giữ
lại mà corpus đã đổi (thêm/bớt tài liệu, đổi tham số chunk), chunk cũ và chunk mới
sẽ nằm lẫn lộn trong cùng collection — retrieval vẫn chạy bình thường nhưng thỉnh
thoảng trả về nội dung từ phiên bản dữ liệu đã bỏ đi, rất khó lần ra nguyên nhân.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# =============================================================================
# CONFIGURATION
# =============================================================================

# Chunking: 800 ký tự, overlap 100 (đúng tham số bài lab yêu cầu).
# Vì sao 800 hợp với văn bản luật: đơn vị trả lời tự nhiên ở đây là một "khoản"
# hoặc một "điều" ngắn. Đo trên corpus thực tế, phần lớn khoản dài 300-700 ký tự,
# nên 800 đủ để giữ trọn một khoản trong cùng một chunk. Cắt nhỏ hơn (500) hay
# làm đứt đôi điều kiện: "Không quá 60 ngày đối với công việc có chức danh nghề
# nghiệp cần trình độ..." bị tách khỏi "Điều 25. Thời gian thử việc", và chunk
# đó mất luôn ngữ cảnh cho biết 60 ngày này nói về cái gì.
# Overlap 100 để câu bị cắt ngang ranh giới vẫn còn nguyên vẹn ở một trong hai
# chunk kề nhau.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# BAAI/bge-m3: multilingual, huấn luyện có tiếng Việt, độ dài ngữ cảnh 8192.
# Lưu ý dung lượng: model ~2.2GB, cộng torch nữa là khoảng 3GB phải tải về.
# Đặt biến môi trường EMBEDDING_MODEL để thử nhanh bằng model nhẹ hơn, ví dụ
# intfloat/multilingual-e5-small (~470MB) khi mạng chậm.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = 1024

VECTOR_STORE = "chromadb"  # "chromadb" | "weaviate" | "faiss"
# Đổi tên theo chủ đề của nhóm (Luật Lao động), không dùng lại tên mẫu của
# starter. Tên collection sai chủ đề dễ khiến người sau tưởng đang đọc nhầm dữ
# liệu khi debug.
COLLECTION_NAME = "labor_law_docs"

# Chroma giới hạn số bản ghi mỗi lần upsert; 500 vừa an toàn vừa đủ nhanh
UPSERT_BATCH_SIZE = 500
EMBED_BATCH_SIZE = 16

# Bỏ chunk quá ngắn. Splitter thỉnh thoảng đẻ ra mảnh vụn chỉ gồm một chữ số
# (sót lại từ bảng biểu) hoặc dòng ký tên cuối văn bản. Những mảnh này vẫn được
# embed và index như chunk bình thường, rồi nổi lên trong kết quả của các truy
# vấn có chứa số — vừa tốn chỗ trong top_k vừa không mang thông tin nào.
MIN_CHUNK_CHARS = 50


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    if not STANDARDIZED_DIR.exists():
        return []

    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        if not content.strip():
            continue

        # Phân loại theo thư mục cha, không phải theo tên file: metadata này để
        # app.py lọc nguồn ("chỉ tra văn bản luật") và để báo cáo đánh giá tách
        # được chất lượng retrieval trên luật gốc so với bài tư vấn.
        doc_type = "legal" if "legal" in md_file.parts else "news"

        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type},
        })

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Cắt documents thành chunk theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Thứ tự separator quyết định chất lượng chunk. Ưu tiên cắt ở ranh giới đoạn
    # ("\n\n") rồi mới tới dòng, rồi mới tới câu. Với văn bản luật, mỗi khoản nằm
    # trên một dòng riêng nên cắt theo dòng gần như luôn trùng ranh giới khoản —
    # chỉ khi khoản dài quá 800 ký tự mới phải cắt sâu hơn vào giữa câu.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    for doc in documents:
        for i, piece in enumerate(splitter.split_text(doc["content"])):
            if len(piece.strip()) < MIN_CHUNK_CHARS:
                continue
            chunks.append({
                "content": piece,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })

    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["content"] for c in chunks]
    # normalize_embeddings=True để vector có độ dài 1. Khi đó cosine similarity
    # bằng đúng tích vô hướng, và khoảng cách cosine của Chroma nằm gọn trong
    # [0, 2] — Task 5 đổi ngược về [0, 1] mới có nghĩa để so ngưỡng ở Task 9.
    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()

    return chunks


def get_collection(reset: bool = False):
    """Lấy (hoặc tạo mới) collection ChromaDB."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"  ✓ Đã xoá collection cũ: {COLLECTION_NAME}")
        except Exception:
            pass  # chưa tồn tại thì thôi

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        # Bắt buộc khai báo cosine. Mặc định của Chroma là L2 (khoảng cách
        # Euclid); nếu để mặc định thì công thức đổi điểm ở Task 5
        # (similarity = 1 - distance) sai hoàn toàn về mặt thang đo, và ngưỡng
        # fallback 0.48 ở Task 9 trở nên vô nghĩa.
        metadata={"hnsw:space": "cosine"},
    )


def index_to_vectorstore(chunks: list[dict], reset: bool = True) -> None:
    """Lưu chunks vào ChromaDB."""
    collection = get_collection(reset=reset)

    ids = [
        f"{c['metadata']['source']}::chunk_{c['metadata']['chunk_index']}"
        for c in chunks
    ]
    documents = [c["content"] for c in chunks]
    embeddings = [c["embedding"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    for start in range(0, len(chunks), UPSERT_BATCH_SIZE):
        end = start + UPSERT_BATCH_SIZE
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  → đã ghi {min(end, len(chunks))}/{len(chunks)} chunk")


def run_pipeline(reset: bool = True) -> None:
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 66)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking : {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Store    : {VECTOR_STORE} → {CHROMA_DIR}")
    print("=" * 66)

    docs = load_documents()
    print(f"\n✓ Đã nạp {len(docs)} tài liệu")
    if not docs:
        print("⚠ Không có tài liệu nào trong data/standardized/ — chạy Task 1-3 trước")
        return

    n_legal = sum(1 for d in docs if d["metadata"]["type"] == "legal")
    print(f"  ({n_legal} văn bản luật, {len(docs) - n_legal} bài tư vấn)")

    chunks = chunk_documents(docs)
    print(f"✓ Đã cắt {len(chunks)} chunk")
    if chunks:
        lengths = [len(c["content"]) for c in chunks]
        print(f"  độ dài: min={min(lengths)}, "
              f"trung bình={sum(lengths) // len(lengths)}, max={max(lengths)}")

    print(f"\nĐang embed bằng {EMBEDDING_MODEL} (lần đầu sẽ phải tải model)...")
    chunks = embed_chunks(chunks)
    print(f"✓ Đã embed {len(chunks)} chunk")

    print("\nĐang ghi vào ChromaDB...")
    index_to_vectorstore(chunks, reset=reset)

    collection = get_collection(reset=False)
    print(f"\n✓ Hoàn tất — collection '{COLLECTION_NAME}' hiện có "
          f"{collection.count()} chunk")


if __name__ == "__main__":
    run_pipeline(reset="--no-reset" not in sys.argv)
