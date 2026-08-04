"""
RAG Chatbot — Trợ Lý Hỏi Đáp Luật Lao Động Cho Người Trẻ.

Nối UI Streamlit vào pipeline: Task 9 (Hybrid Retrieval + RRF + PageIndex
fallback) → Task 10 (Generation có citation).

Chạy:
    streamlit run app.py
"""

import html
import re
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Trợ Lý Luật Lao Động",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .src-card {
        border-left: 3px solid #4c8bf5;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.7rem;
        background: rgba(76, 139, 245, 0.06);
        border-radius: 0 6px 6px 0;
    }
    .src-head { font-size: 0.86rem; opacity: 0.85; margin-bottom: 0.35rem; }
    .src-body { font-size: 0.88rem; line-height: 1.55; }
    .src-body mark {
        background: #ffe27a; color: #222; padding: 0 2px; border-radius: 2px;
    }
    .pill {
        display: inline-block; padding: 1px 8px; border-radius: 10px;
        font-size: 0.74rem; margin-right: 5px;
    }
    .pill-legal { background: #2e7d32; color: #fff; }
    .pill-news  { background: #6a1b9a; color: #fff; }
    .pill-score { background: #37474f; color: #fff; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPERS
# =============================================================================

_STOPWORDS = {
    "là", "của", "và", "có", "cho", "được", "thì", "khi", "không", "phải",
    "tôi", "bao", "nhiêu", "gì", "nào", "với", "trong", "một", "các", "này",
    "mà", "để", "ở", "về", "hay", "bằng", "sau", "trước", "đã", "sẽ", "còn",
}


def _keywords(query: str) -> list[str]:
    """Từ khoá dùng để bôi vàng trong đoạn trích nguồn."""
    tokens = re.findall(r"\w+", query.lower(), flags=re.UNICODE)
    kws = {t for t in tokens if len(t) > 2 and t not in _STOPWORDS}
    # Giữ cả số hiệu điều luật kiểu "25", "145" — rất quan trọng với văn bản luật
    kws |= set(re.findall(r"\d+", query))
    return sorted(kws, key=len, reverse=True)[:12]


def _highlight(text: str, keywords: list[str]) -> str:
    """Escape HTML rồi bọc <mark> quanh từ khoá."""
    out = html.escape(text)
    for kw in keywords:
        out = re.sub(f"({re.escape(html.escape(kw))})", r"<mark>\1</mark>",
                     out, flags=re.IGNORECASE)
    return out


def render_sources(sources: list[dict], query: str, key_prefix: str) -> None:
    """Vùng hiển thị danh sách tài liệu tham khảo."""
    if not sources:
        return

    kws = _keywords(query)
    with st.expander(f"📚 Nguồn tham khảo ({len(sources)} đoạn)", expanded=False):
        for i, src in enumerate(sources, 1):
            meta = src.get("metadata", {}) or {}
            name = meta.get("source", "không rõ nguồn")
            doc_type = meta.get("type", "")
            score = float(src.get("score", 0.0))
            origin = src.get("source", "")

            pill_cls = "pill-legal" if doc_type == "legal" else "pill-news"
            pill_txt = "văn bản luật" if doc_type == "legal" else "bài tư vấn"

            content = " ".join((src.get("content") or "").split())
            snippet = content[:600] + ("…" if len(content) > 600 else "")

            st.markdown(
                f"""<div class="src-card">
                  <div class="src-head">
                    <b>[{i}] {html.escape(name)}</b>
                    <span class="pill {pill_cls}">{pill_txt}</span>
                    <span class="pill pill-score">score {score:.4f}</span>
                    <span class="pill pill-score">{html.escape(origin)}</span>
                  </div>
                  <div class="src-body">{_highlight(snippet, kws)}</div>
                </div>""",
                unsafe_allow_html=True,
            )


def render_diagnostics(diag: dict) -> None:
    """Hiển thị chẩn đoán retrieval — cho thấy fallback có kích hoạt hay không."""
    if not diag:
        return
    with st.expander("🔍 Chẩn đoán retrieval", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Dense hits", diag.get("n_semantic", 0))
        c2.metric("BM25 hits", diag.get("n_lexical", 0))
        c3.metric("Cosine tốt nhất", f"{diag.get('best_cosine', 0):.4f}")
        c4.metric("Ngưỡng fallback", diag.get("threshold", "—"))

        if diag.get("used_pageindex"):
            st.warning("Cosine dưới ngưỡng → đã chuyển sang **PageIndex vectorless**.")
        elif diag.get("below_threshold"):
            st.warning(
                "Cosine dưới ngưỡng nhưng PageIndex chưa cấu hình được — "
                "đang trả kết quả hybrid điểm thấp. Câu trả lời có thể không đủ căn cứ."
            )
        else:
            st.success("Cosine trên ngưỡng → dùng kết quả Hybrid (Dense + BM25 + RRF).")


def build_search_query(query: str, history: list[dict], enabled: bool) -> str:
    """
    Ghép ngữ cảnh hội thoại vào truy vấn khi câu hỏi là follow-up.

    Câu follow-up thường rất ngắn và chứa đại từ thay thế ("vậy còn...", "thế
    trường hợp đó thì sao"). Đem nguyên văn đi retrieve thì gần như chắc chắn
    trượt, vì bản thân nó không mang từ khoá pháp lý nào. Cách xử lý ở đây là
    ghép thêm câu hỏi người dùng gần nhất để khôi phục ngữ cảnh — rẻ và không
    tốn thêm lượt gọi LLM (khác với cách dùng LLM để viết lại câu hỏi).
    """
    if not enabled or not history:
        return query

    followup_markers = ("vậy", "thế", "còn", "đó", "này", "trên", "vừa rồi", "nó")
    is_short = len(query.split()) <= 10
    has_marker = any(m in query.lower() for m in followup_markers)
    if not (is_short and has_marker):
        return query

    prev = [m["content"] for m in history if m["role"] == "user"]
    if not prev:
        return query
    return f"{prev[-1]} {query}"


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("⚖️ Trợ Lý Luật Lao Động")
    st.caption(
        "Tra cứu quy định về thử việc, làm thêm giờ, nghỉ phép, hợp đồng và "
        "sa thải — dành cho người đi làm trẻ."
    )

    st.divider()
    st.subheader("💡 Câu hỏi gợi ý")
    suggestions = [
        "Thời gian thử việc tối đa cho vị trí lập trình viên là bao lâu và lương thử việc tối thiểu bằng bao nhiêu % lương chính thức?",
        "Công ty sa thải tôi qua tin nhắn Zalo mà không báo trước 30 ngày thì có đúng luật không?",
        "Một tháng tôi bị bắt OT tối đa bao nhiêu giờ?",
        "Làm thêm giờ ngày lễ được trả lương bao nhiêu phần trăm?",
        "Nghỉ việc chưa dùng hết phép năm có được thanh toán tiền không?",
        "Hợp đồng xác định thời hạn được ký tối đa mấy lần?",
    ]
    for i, s in enumerate(suggestions):
        label = s if len(s) <= 58 else s[:55] + "…"
        if st.button(label, use_container_width=True, key=f"sug_{i}"):
            st.session_state["pending_query"] = s

    st.divider()
    st.subheader("⚙️ Thiết lập")
    top_k = st.slider("Số đoạn ngữ cảnh (top_k)", 3, 10, 5,
                      help="Số chunk đưa vào prompt của LLM")
    use_reranking = st.toggle("Gộp bằng RRF (Hybrid)", value=True,
                              help="Tắt để chỉ nối kết quả Dense + BM25, không fuse")
    use_hyde = st.toggle("Bật HyDE", value=False,
                         help="Sinh câu trả lời giả định rồi mới đi tìm — tốn thêm 1 lượt LLM")
    score_threshold = st.slider("Ngưỡng cosine kích hoạt fallback", 0.0, 1.0, 0.48, 0.01,
                                help="So với cosine GỐC của semantic search, không phải điểm RRF")
    use_memory = st.toggle("Ghi nhớ hội thoại", value=True,
                           help="Ghép ngữ cảnh câu trước khi bạn hỏi tiếp kiểu 'vậy còn…'")

    st.divider()
    if st.button("🗑️ Xoá lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("**Kiến trúc:**")
    st.caption(
        "Dense (bge-m3 / ChromaDB) + BM25 → RRF (k=60) → "
        "PageIndex fallback khi cosine < ngưỡng → LLM sinh câu trả lời có trích dẫn"
    )
    st.caption("**Nguồn:** Bộ luật Lao động 2019, NĐ 145/2020, NĐ 12/2022, NĐ 293/2025")

    st.warning(
        "Thông tin mang tính tham khảo, không thay thế tư vấn pháp lý chính thức.",
        icon="⚠️",
    )


# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# =============================================================================
# MAIN
# =============================================================================

st.title("⚖️ Trợ Lý Hỏi Đáp Luật Lao Động")
st.caption(
    "Hỏi về thử việc, OT, nghỉ phép, hợp đồng, sa thải — câu trả lời luôn kèm "
    "trích dẫn điều luật gốc."
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []), msg.get("query", ""),
                           key_prefix=f"hist_{id(msg)}")
            render_diagnostics(msg.get("diagnostics", {}))


user_input = st.chat_input("Nhập câu hỏi về luật lao động…")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None

    history = list(st.session_state.messages)
    search_query = build_search_query(query, history, use_memory)

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
        if search_query != query:
            st.caption(f"↳ đã ghép ngữ cảnh để tìm kiếm: *{search_query}*")

    with st.chat_message("assistant"):
        answer, sources, diagnostics = "", [], {}

        with st.spinner("Đang tra cứu văn bản pháp luật…"):
            try:
                from src.task9_retrieval_pipeline import retrieve_verbose
                from src.task10_generation import generate_with_citation

                out = retrieve_verbose(
                    search_query,
                    top_k=top_k,
                    score_threshold=score_threshold,
                    use_reranking=use_reranking,
                    use_hyde=use_hyde,
                )
                sources = out["results"]
                diagnostics = out["diagnostics"]

                response = generate_with_citation(query, context_chunks=sources)
                answer = response.get("answer", "")

            except NotImplementedError:
                answer = ("⚠️ **Pipeline chưa hoàn thiện.** Kiểm tra lại "
                          "`src/task9_retrieval_pipeline.py` và `src/task10_generation.py`.")
            except Exception as e:
                answer = (f"❌ **Lỗi khi chạy RAG pipeline:** `{type(e).__name__}: {e}`\n\n"
                          "Kiểm tra: đã chạy `python -m src.task4_chunking_indexing` để "
                          "index ChromaDB chưa, và `OPENROUTER_API_KEY` trong `.env` "
                          "còn hiệu lực không.")

        st.markdown(answer or "*(không có nội dung trả về)*")
        render_sources(sources, query, key_prefix="live")
        render_diagnostics(diagnostics)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "diagnostics": diagnostics,
        "query": query,
    })
