"""
Task 10 — Generation Có Citation.

Luồng: retrieve (Task 9) → reorder chống lost-in-the-middle → format context có
nguồn → gọi LLM → câu trả lời kèm trích dẫn.

LLM: OpenRouter (dùng chung interface với OpenAI SDK). Nhiều model hậu tố ":free"
không tính phí — https://openrouter.ai/models?max_price=0

Chạy:
    python -m src.task10_generation
"""

import os

from dotenv import load_dotenv

load_dotenv()

from src.task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION
# =============================================================================

# Số chunk đưa vào context. 5 là điểm cân bằng: đủ dẫn chứng để trả lời một câu
# hỏi pháp luật (thường cần cả điều luật gốc lẫn nghị định hướng dẫn), nhưng
# chưa dài tới mức LLM bắt đầu bỏ sót phần giữa.
TOP_K = 5

# top_p = 0.9: cắt bớt đuôi phân phối, loại các token rất ít khả năng. Với hỏi
# đáp pháp luật, phần đuôi đó chính là chỗ mô hình bịa ra số điều luật không tồn
# tại — cắt đi thì an toàn hơn mà không làm câu văn cứng.
TOP_P = 0.9

# temperature = 0.1: gần như tất định. Đây là tra cứu pháp luật, không phải viết
# sáng tạo — cùng một câu hỏi phải cho cùng một câu trả lời. Skeleton để 0.3,
# hạ xuống 0.1 vì mỗi độ ngẫu nhiên thêm vào đều là thêm rủi ro sai điều luật.
TEMPERATURE = 0.1

MAX_TOKENS = 1000

# Model mặc định :free để cả nhóm chạy được khi chưa có credit.
# Đã thử qua 7 model free trên OpenRouter (04/08/2026):
#   - llama-3.3-70b:free       → 404, không còn miễn phí
#   - gemma-4-31b-it:free      → 429, provider quá tải
#   - nemotron-3-super-120b    → chạy được nhưng in cả chuỗi suy luận ra output
#   - ling-3.0-flash, laguna-s, gpt-oss-20b → trả content rỗng
#   - gemma-4-26b-a4b-it:free  → sạch, tiếng Việt tốt, trả lời đúng ✓
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemma-4-26b-a4b-it:free")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

NO_EVIDENCE_MESSAGE = (
    "Tôi không thể xác minh thông tin này từ các văn bản pháp luật hiện có "
    "trong cơ sở dữ liệu."
)


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Bạn là trợ lý hỏi đáp pháp luật lao động Việt Nam, phục vụ
người lao động trẻ (thử việc, làm thêm giờ, nghỉ phép, hợp đồng, sa thải).

QUY TẮC BẮT BUỘC:

1. CHỈ trả lời dựa trên phần NGỮ CẢNH được cung cấp bên dưới. Không dùng kiến
   thức bên ngoài, kể cả khi bạn chắc chắn về nó.

2. Sau mỗi khẳng định, chèn ngay trích dẫn dạng [Tên nguồn, Điều/Khoản]:
       "Thời gian thử việc không quá 60 ngày đối với công việc cần trình độ cao
       đẳng trở lên [Bộ luật Lao động 2019, Điều 25]."

3. Nếu ngữ cảnh KHÔNG chứa đủ căn cứ, trả lời đúng câu sau và không suy đoán:
       "{no_evidence}"

4. Không bịa số hiệu điều luật, nghị định hay mức phạt. Chỉ nêu con số xuất hiện
   trong ngữ cảnh.

5. Trả lời bằng tiếng Việt, ngắn gọn, dễ hiểu với người không học luật. Giải
   thích thuật ngữ pháp lý khi lần đầu dùng.

6. Nếu vấn đề còn phụ thuộc tình tiết cụ thể, nói rõ điều đó thay vì kết luận
   chắc chắn."""


# =============================================================================
# Reordering — chống "lost in the middle"
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp lại chunk: quan trọng nhất ở ĐẦU và CUỐI, ít quan trọng vào GIỮA.

    Cơ sở: Liu et al. (2023), "Lost in the Middle". LLM nhớ tốt thông tin ở đầu
    và cuối context, còn phần giữa thì độ chính xác tụt rõ rệt — đường cong hình
    chữ U. Nếu cứ xếp theo thứ hạng giảm dần thì chunk hạng 2, 3 (vẫn rất liên
    quan) lại rơi đúng vào vùng mù đó.

    Cách xếp: [1, 3, 5, 4, 2] thay vì [1, 2, 3, 4, 5]
        - chunk lẻ (0,2,4...) xếp xuôi ở nửa đầu
        - chunk chẵn (1,3...) xếp ngược ở nửa sau
    Nhờ vậy hạng 1 nằm đầu, hạng 2 nằm cuối — hai vị trí LLM đọc kỹ nhất.

    Args:
        chunks: List chunk đã sắp theo độ liên quan giảm dần

    Returns:
        List cùng độ dài, đã đảo vị trí.
    """
    if len(chunks) <= 2:
        return list(chunks)

    front = chunks[0::2]          # hạng 1, 3, 5, ...
    back = chunks[1::2][::-1]     # hạng ..., 4, 2
    return front + back


# =============================================================================
# Format context
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Ghép chunk thành khối ngữ cảnh, mỗi khối gắn nhãn nguồn.

    Nhãn nguồn phải nằm NGAY CẠNH nội dung, không gom thành danh sách ở cuối:
    LLM cần biết câu nào thuộc văn bản nào ngay tại chỗ nó đang đọc thì mới trích
    dẫn đúng. Gom nguồn ra chỗ khác là mời mô hình gán nhầm trích dẫn.
    """
    if not chunks:
        return ""

    blocks = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {}) or {}
        source = meta.get("source", "không rõ nguồn")
        section = meta.get("section", "")
        doc_type = meta.get("type", "")

        label = f"[Nguồn {i}: {source}"
        if section:
            label += f" — {section}"
        if doc_type:
            label += f" ({'văn bản luật' if doc_type == 'legal' else 'bài tư vấn'})"
        label += "]"

        blocks.append(f"{label}\n{chunk.get('content', '').strip()}")

    return "\n\n---\n\n".join(blocks)


# =============================================================================
# Gọi LLM
# =============================================================================

def _get_llm_client():
    """Tạo client LLM. Ưu tiên OpenRouter, lùi về OpenAI."""
    from openai import OpenAI

    if OPENROUTER_API_KEY:
        return OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        ), LLM_MODEL

    if OPENAI_API_KEY:
        return OpenAI(api_key=OPENAI_API_KEY), os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    raise RuntimeError(
        "Chưa có OPENROUTER_API_KEY (hoặc OPENAI_API_KEY) trong .env — "
        "không gọi được LLM. Tạo file .env từ .env.example và điền key."
    )


def generate_with_citation(
    query: str,
    context_chunks: list[dict] | None = None,
    top_k: int = TOP_K,
) -> dict:
    """
    Sinh câu trả lời có trích dẫn nguồn.

    Args:
        query: Câu hỏi
        context_chunks: Chunk ngữ cảnh. Bỏ trống thì tự gọi retrieve() (Task 9).
        top_k: Số chunk dùng khi phải tự retrieve

    Returns:
        {
            'answer': str,
            'sources': list[dict],      # chunk đã dùng, theo thứ tự liên quan
            'context_used': int,
            'model': str,
            'reordered': bool
        }
    """
    if context_chunks is None:
        context_chunks = retrieve(query, top_k=top_k)

    if not context_chunks:
        return {
            "answer": NO_EVIDENCE_MESSAGE,
            "sources": [],
            "context_used": 0,
            "model": None,
            "reordered": False,
        }

    # Đảo vị trí TRƯỚC khi ghép prompt, nhưng 'sources' trả về vẫn giữ thứ tự
    # liên quan giảm dần — người dùng đọc danh sách nguồn muốn thấy cái liên
    # quan nhất đứng đầu, còn thứ tự đảo chỉ phục vụ cách LLM đọc context.
    reordered = reorder_for_llm(context_chunks)
    context = format_context(reordered)

    client, model = _get_llm_client()

    user_prompt = f"""NGỮ CẢNH:

{context}

---

CÂU HỎI: {query}

Trả lời dựa trên ngữ cảnh trên, kèm trích dẫn nguồn cho từng khẳng định."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": SYSTEM_PROMPT.format(no_evidence=NO_EVIDENCE_MESSAGE)},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
    )

    answer = (response.choices[0].message.content or "").strip()

    return {
        "answer": answer or NO_EVIDENCE_MESSAGE,
        "sources": context_chunks,
        "context_used": len(context_chunks),
        "model": model,
        "reordered": True,
    }


if __name__ == "__main__":
    print("=" * 66)
    print("Task 10: Generation có Citation")
    print("=" * 66)

    # --- Minh hoạ reorder (không cần API key) ---
    demo = [{"content": f"Chunk hạng {i}", "score": 1.0 - i * 0.1} for i in range(5)]
    print("\nReorder chống lost-in-the-middle:")
    print(f"  trước: {[c['content'] for c in demo]}")
    print(f"  sau  : {[c['content'] for c in reorder_for_llm(demo)]}")
    print("  → hạng 1 ở đầu, hạng 2 ở cuối: hai vị trí LLM đọc kỹ nhất")

    if not (OPENROUTER_API_KEY or OPENAI_API_KEY):
        print("\n⚠ Chưa có OPENROUTER_API_KEY trong .env — bỏ qua phần gọi LLM.")
        print("  Tạo .env từ .env.example rồi điền key để chạy đầy đủ.")
        raise SystemExit(0)

    for question in [
        "Thời gian thử việc tối đa cho vị trí lập trình viên là bao lâu và "
        "lương thử việc tối thiểu bằng bao nhiêu % lương chính thức?",
        "Công ty sa thải tôi qua tin nhắn Zalo mà không báo trước 30 ngày "
        "thì có đúng luật không?",
    ]:
        print(f"\n{'─' * 66}")
        print(f"Hỏi: {question}")
        result = generate_with_citation(question)
        print(f"\n{result['answer']}")
        print(f"\n[dùng {result['context_used']} nguồn, model {result['model']}]")
