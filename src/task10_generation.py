"""
Task 10 — Generation Có Citation.
"""

from __future__ import annotations

import re
import os
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3

LLM_MODEL = "openai/gpt-4o-mini"


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi về dịch vụ và chính sách đại học
(học phí, học bổng, ký túc xá, thư viện, đăng ký học phần).

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin từ context được cung cấp — KHÔNG bịa đặt
2. Mỗi khẳng định phải có trích dẫn ngay sau, ví dụ: [Tuition Fees, 2026]
3. Nếu context không đủ thông tin → trả lời: "Tôi không thể xác minh thông tin này từ nguồn hiện có"
4. Trả lời bằng tiếng Việt, có cấu trúc rõ ràng theo đoạn văn
5. Không suy luận hay mở rộng ngoài những gì được nêu trong context"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return list(chunks)

    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {}) or {}
        source = metadata.get("source", f"Source {index}")
        doc_type = metadata.get("type", "unknown")
        context_parts.append(
            f"[Document {index} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk.get('content', '')}\n"
        )
    return "\n---\n".join(context_parts)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[\w]+", str(text).lower(), flags=re.UNICODE))


def _extractive_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    query_tokens = _tokenize(query)
    sentences: list[tuple[float, str, str]] = []
    for chunk in chunks:
        content = str(chunk.get("content", ""))
        source = chunk.get("metadata", {}).get("source", "Unknown")
        for sentence in re.split(r"(?<=[.!?])\s+", content):
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_tokens = _tokenize(sentence)
            if not sentence_tokens:
                continue
            score = len(query_tokens & sentence_tokens) / max(len(query_tokens), 1)
            if score > 0:
                sentences.append((score, sentence, source))

    if not sentences:
        chunk = chunks[0]
        source = chunk.get("metadata", {}).get("source", "Unknown")
        snippet = str(chunk.get("content", "")).strip().splitlines()[0][:300]
        return f"Dựa trên nguồn {source}, thông tin liên quan nhất hiện có là: {snippet} [{source}]"

    sentences.sort(key=lambda item: item[0], reverse=True)
    best_sentence = sentences[0][1]
    source = sentences[0][2]
    return f"Dựa trên nguồn {source}, {best_sentence} [{source}]"


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            answer = response.choices[0].message.content or ""
        except Exception:
            answer = _extractive_answer(query, reordered)
    else:
        answer = _extractive_answer(query, reordered)

    return {
        "answer": answer,
        "sources": reordered,
        "retrieval_source": reordered[0].get("source", "hybrid") if reordered else "none",
        "context": context,
    }


if __name__ == "__main__":
    test_queries = [
        "Học phí tại RMIT Vietnam là bao nhiêu?",
        "Làm sao để đặt phòng học nhóm ở thư viện?",
        "Sinh viên quốc tế có những học bổng nào?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
