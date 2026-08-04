"""
Task 8 — PageIndex Vectorless RAG.
"""

from __future__ import annotations

import re
import os
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[\w]+", str(text).lower(), flags=re.UNICODE))


def _split_sections(markdown: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_title = "Document"
    buffer: list[str] = []

    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            if buffer:
                sections.append((current_title, "\n".join(buffer).strip()))
                buffer = []
            current_title = stripped.lstrip("#").strip() or current_title
        else:
            buffer.append(line)

    if buffer:
        sections.append((current_title, "\n".join(buffer).strip()))
    return [(title, content) for title, content in sections if content]


@lru_cache(maxsize=1)
def _load_sections() -> list[dict]:
    sections: list[dict] = []
    if not STANDARDIZED_DIR.exists():
        return sections

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception:
            continue
        for section_title, section_content in _split_sections(content):
            sections.append(
                {
                    "content": section_content,
                    "metadata": {
                        "source": md_file.name,
                        "type": md_file.parent.name,
                        "section": section_title,
                        "source_path": md_file.relative_to(STANDARDIZED_DIR).as_posix(),
                    },
                }
            )
    return sections


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    return [str(md_file) for md_file in STANDARDIZED_DIR.rglob("*.md")]


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if not isinstance(query, str):
        raise TypeError("query phải là string")
    query = query.strip()
    if not query or top_k <= 0:
        return []

    sections = _load_sections()
    if not sections:
        return []

    query_tokens = _tokenize(query)
    query_lower = query.lower()
    scored: list[dict] = []
    for section in sections:
        content = str(section.get("content", ""))
        content_tokens = _tokenize(content)
        if not content_tokens:
            continue

        overlap = len(query_tokens & content_tokens)
        exact_phrase_bonus = 1.0 if query_lower in content.lower() else 0.0
        heading_bonus = 0.0
        section_title = str(section.get("metadata", {}).get("section", "")).lower()
        if query_tokens & _tokenize(section_title):
            heading_bonus = 0.5

        score = overlap + exact_phrase_bonus + heading_bonus
        if score <= 0:
            continue

        scored.append(
            {
                "content": content,
                "score": round(float(score), 6),
                "metadata": dict(section.get("metadata") or {}),
                "source": "pageindex",
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    if scored:
        return scored[:top_k]

    # Fallback cuối cùng: lấy section đầu tiên để đảm bảo pipeline không crash.
    first = sections[0]
    return [
        {
            "content": str(first.get("content", "")),
            "score": 0.0,
            "metadata": dict(first.get("metadata") or {}),
            "source": "pageindex",
        }
    ]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("tuition fee payment methods", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
