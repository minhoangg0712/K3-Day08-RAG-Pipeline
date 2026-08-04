"""
Task 4 — Chunking & indexing tài liệu luật lao động vào ChromaDB.

Pipeline:
    data/standardized/**/*.md
        -> load_documents()
        -> chunk_documents()
        -> embed_chunks()
        -> index_to_vectorstore()

Chạy từ thư mục gốc của repository:
    python -m src.task4_chunking_indexing

Lần chạy mặc định sẽ tạo lại collection để bảo đảm chunk cũ không bị trộn với
corpus hiện tại. Dùng ``--no-reset`` nếu chỉ muốn upsert lại các tài liệu đang có.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


PROJECT_DIR = Path(__file__).resolve().parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"
CHROMA_DIR = PROJECT_DIR / "chroma_db"


# =============================================================================
# CONFIGURATION
# =============================================================================

# RecursiveCharacterTextSplitter được chọn vì corpus gồm cả văn bản luật và bài
# hướng dẫn Markdown. Splitter ưu tiên ranh giới đoạn/dòng/câu trước khi phải cắt
# cứng, nên giữ được cấu trúc Điều/Khoản tốt hơn một cửa sổ ký tự thuần túy.
# 500 ký tự đủ ngắn cho retrieval chính xác; overlap 50 (10%) giữ lại ngữ cảnh ở
# biên chunk mà không tạo quá nhiều nội dung trùng lặp.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

# BAAI/bge-m3 tạo vector 1024 chiều, hỗ trợ đa ngôn ngữ và phù hợp corpus trộn
# tiếng Việt với thuật ngữ tiếng Anh. normalize_embeddings=True được dùng để
# cosine distance trong Chroma có ý nghĩa nhất quán.
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
EMBEDDING_BATCH_SIZE = 16

VECTOR_STORE = "chromadb"
COLLECTION_NAME = "vietnam_labor_law_docs"
CHROMA_BATCH_SIZE = 256

_ALLOWED_METADATA_TYPES = (str, int, float, bool)


# =============================================================================
# DOCUMENT LOADING
# =============================================================================

def _markdown_title(content: str, fallback: str) -> str:
    """Lấy heading H1 đầu tiên làm tiêu đề tài liệu."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            if title:
                return title
    return fallback.replace("-", " ").replace("_", " ").strip().title()


def _document_type(relative_path: Path) -> str:
    """Suy ra loại tài liệu từ thư mục con đầu tiên."""
    if len(relative_path.parts) > 1:
        return relative_path.parts[0].lower()
    return "unknown"


def load_documents(directory: Path | str = STANDARDIZED_DIR) -> list[dict]:
    """
    Đọc các file Markdown không rỗng từ ``data/standardized``.

    Args:
        directory: Cho phép truyền thư mục khác khi test hoặc tái sử dụng.

    Returns:
        List gồm ``content`` và metadata scalar tương thích ChromaDB.
    """
    source_dir = Path(directory)
    if not source_dir.exists():
        return []
    if not source_dir.is_dir():
        raise ValueError(f"Đường dẫn standardized không phải thư mục: {source_dir}")

    documents: list[dict] = []
    for md_file in sorted(source_dir.rglob("*.md")):
        if not md_file.is_file() or md_file.name.startswith("."):
            continue

        try:
            content = md_file.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError as exc:
            raise ValueError(f"File không phải UTF-8: {md_file}") from exc

        # Không index file rỗng vì ChromaDB và embedding model không nhận được
        # thêm tín hiệu hữu ích từ chúng.
        if not content:
            continue

        relative_path = md_file.relative_to(source_dir)
        source_path = relative_path.as_posix()
        document_id = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:16]
        documents.append(
            {
                "content": content,
                "metadata": {
                    "document_id": document_id,
                    "source": md_file.name,
                    "source_path": source_path,
                    "title": _markdown_title(content, md_file.stem),
                    "type": _document_type(relative_path),
                    "language": "vi",
                },
            }
        )

    return documents


# =============================================================================
# CHUNKING
# =============================================================================

def _fallback_split_text(text: str) -> list[str]:
    """Fallback nhỏ gọn để test được khi chưa cài langchain-text-splitters."""
    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        hard_end = min(start + CHUNK_SIZE, text_length)
        end = hard_end

        if hard_end < text_length:
            search_start = start + int(CHUNK_SIZE * 0.6)
            candidates = [
                text.rfind("\n\n", search_start, hard_end),
                text.rfind("\n", search_start, hard_end),
                text.rfind(". ", search_start, hard_end),
                text.rfind(" ", search_start, hard_end),
            ]
            boundary = max(candidates)
            if boundary > start:
                end = boundary + (2 if text[boundary : boundary + 2] == ". " else 0)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break

        next_start = end - CHUNK_OVERLAP
        start = next_start if next_start > start else end

    return chunks


def _build_text_splitter():
    """Khởi tạo splitter chính; trả về None nếu dependency chưa được cài."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        return None

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", "; ", " ", ""],
        keep_separator=True,
        add_start_index=True,
        strip_whitespace=True,
        length_function=len,
    )


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chia tài liệu thành các chunk và bảo toàn metadata nguồn.

    Mỗi chunk được bổ sung ``chunk_index``, ``start_index`` và ``chunk_id``.
    ``chunk_id`` ổn định theo tài liệu, vị trí và nội dung nên có thể dùng làm ID
    khi upsert vào ChromaDB.
    """
    if not isinstance(documents, list):
        raise TypeError("documents phải là list[dict]")

    splitter = _build_text_splitter()
    chunks: list[dict] = []

    for doc in documents:
        if not isinstance(doc, dict):
            raise TypeError("Mỗi document phải là dict")

        content = str(doc.get("content", "")).strip()
        if not content:
            continue
        metadata = dict(doc.get("metadata") or {})
        document_id = str(
            metadata.get("document_id")
            or hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        )
        metadata["document_id"] = document_id

        if splitter is not None:
            split_docs = splitter.create_documents([content], metadatas=[metadata])
            split_items = [
                (item.page_content.strip(), int(item.metadata.get("start_index", 0)))
                for item in split_docs
                if item.page_content.strip()
            ]
        else:
            fallback_chunks = _fallback_split_text(content)
            split_items = []
            search_from = 0
            for chunk_text in fallback_chunks:
                start_index = content.find(chunk_text, search_from)
                if start_index < 0:
                    start_index = content.find(chunk_text)
                if start_index < 0:
                    start_index = search_from
                split_items.append((chunk_text, start_index))
                search_from = max(start_index + len(chunk_text) - CHUNK_OVERLAP, 0)

        for chunk_index, (chunk_text, start_index) in enumerate(split_items):
            content_digest = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:12]
            chunk_id = f"{document_id}-{chunk_index:05d}-{content_digest}"
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_index,
                        "start_index": start_index,
                        "char_count": len(chunk_text),
                    },
                }
            )

    return chunks


# =============================================================================
# EMBEDDING
# =============================================================================

@lru_cache(maxsize=1)
def get_embedding_model():
    """Tải một lần và cache SentenceTransformer trong suốt process."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu sentence-transformers. Chạy: "
            "python -m pip install sentence-transformers"
        ) from exc

    return SentenceTransformer(EMBEDDING_MODEL)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Sinh normalized dense embedding cho toàn bộ chunk theo batch."""
    if not chunks:
        return []

    texts = [str(chunk.get("content", "")).strip() for chunk in chunks]
    if any(not text for text in texts):
        raise ValueError("Không thể embedding chunk rỗng")

    model = get_embedding_model()
    model_dimension = model.get_sentence_embedding_dimension()
    if model_dimension and model_dimension != EMBEDDING_DIM:
        raise ValueError(
            f"Sai embedding dimension: model trả {model_dimension}, "
            f"cấu hình yêu cầu {EMBEDDING_DIM}"
        )

    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=len(texts) > EMBEDDING_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if len(embeddings) != len(chunks):
        raise RuntimeError("Số embedding trả về không khớp số chunk")

    output: list[dict] = []
    for chunk, embedding in zip(chunks, embeddings):
        item = {**chunk, "metadata": dict(chunk.get("metadata") or {})}
        item["embedding"] = embedding.tolist()
        output.append(item)
    return output


# =============================================================================
# CHROMADB
# =============================================================================

def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Đổi metadata phức tạp sang JSON vì Chroma chỉ nhận scalar."""
    sanitized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, _ALLOWED_METADATA_TYPES):
            sanitized[str(key)] = value
        else:
            sanitized[str(key)] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return sanitized


def get_chroma_client():
    """Mở Chroma persistent client dùng chung cho Task 4 và Task 5."""
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu chromadb. Chạy: python -m pip install chromadb"
        ) from exc

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(*, reset: bool = False):
    """Lấy collection cosine; tùy chọn xóa collection cũ trước khi tạo."""
    client = get_chroma_client()
    if reset:
        collection_names = {
            item if isinstance(item, str) else item.name
            for item in client.list_collections()
        }
        if COLLECTION_NAME in collection_names:
            client.delete_collection(COLLECTION_NAME)

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": EMBEDDING_DIM,
            "domain": "vietnam_labor_law",
        },
    )


def _batched(items: list[dict], batch_size: int) -> Iterable[list[dict]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def index_to_vectorstore(chunks: list[dict], *, reset_collection: bool = False):
    """
    Upsert các chunk đã embedding vào ChromaDB và trả về collection.

    Khi không reset, các chunk cũ thuộc cùng ``source_path`` được xóa trước khi
    upsert để tránh sót chunk nếu tài liệu mới ngắn hơn bản cũ.
    """
    if not chunks:
        raise ValueError("Không có chunk để index")

    for index, chunk in enumerate(chunks):
        embedding = chunk.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise ValueError(f"Chunk {index} chưa có embedding")
        if len(embedding) != EMBEDDING_DIM:
            raise ValueError(
                f"Chunk {index} có embedding dimension {len(embedding)}, "
                f"yêu cầu {EMBEDDING_DIM}"
            )

    collection = get_collection(reset=reset_collection)

    if not reset_collection:
        source_paths = sorted(
            {
                str(chunk.get("metadata", {}).get("source_path", ""))
                for chunk in chunks
                if chunk.get("metadata", {}).get("source_path")
            }
        )
        for source_path in source_paths:
            collection.delete(where={"source_path": source_path})

    for batch in _batched(chunks, CHROMA_BATCH_SIZE):
        ids = [str(item["metadata"]["chunk_id"]) for item in batch]
        collection.upsert(
            ids=ids,
            documents=[str(item["content"]) for item in batch],
            embeddings=[item["embedding"] for item in batch],
            metadatas=[_sanitize_metadata(dict(item["metadata"])) for item in batch],
        )

    return collection


# =============================================================================
# END-TO-END PIPELINE
# =============================================================================

def run_pipeline(*, reset_collection: bool = True):
    """Chạy load -> chunk -> embed -> index và trả về Chroma collection."""
    print("=" * 60)
    print("Task 4: Labor Law Chunking & Indexing")
    print(f"  Chunking : {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Store    : {VECTOR_STORE} -> {CHROMA_DIR}")
    print(f"  Reset    : {reset_collection}")
    print("=" * 60)

    documents = load_documents()
    if not documents:
        raise RuntimeError(f"Không tìm thấy file Markdown có nội dung trong {STANDARDIZED_DIR}")
    print(f"✓ Loaded {len(documents)} documents")

    chunks = chunk_documents(documents)
    if not chunks:
        raise RuntimeError("Không tạo được chunk nào từ corpus")
    print(f"✓ Created {len(chunks)} chunks")

    embedded_chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(embedded_chunks)} chunks")

    collection = index_to_vectorstore(
        embedded_chunks,
        reset_collection=reset_collection,
    )
    print(f"✓ Indexed {collection.count()} chunks to collection '{COLLECTION_NAME}'")
    return collection


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk and index labor-law Markdown files")
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Không xóa toàn bộ collection; chỉ thay chunks của các source hiện tại",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(reset_collection=not args.no_reset)
