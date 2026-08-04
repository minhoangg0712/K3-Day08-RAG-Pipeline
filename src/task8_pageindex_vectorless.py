"""
Task 8 — PageIndex Vectorless RAG (fallback cho hybrid search).

Đăng ký & lấy API key: https://pageindex.ai/
SDK: https://github.com/VectifyAI/PageIndex        pip install pageindex

Vectorless nghĩa là gì và vì sao dùng làm fallback:
    Task 4-6 cắt nhỏ tài liệu rồi tìm theo độ tương đồng. Cách đó hỏng khi câu
    hỏi cần nhìn toàn cục — "nghị định này gồm mấy chương", "thủ tục xử lý kỷ
    luật gồm những bước nào theo thứ tự" — vì không chunk đơn lẻ nào chứa đủ câu
    trả lời, và chọn top_k theo độ tương đồng thì mỗi chunk chỉ là một mảnh rời.

    PageIndex đi theo hướng khác: nó dựng cây mục lục của tài liệu (chương → mục
    → điều) rồi cho LLM duyệt cây đó như người tra cứu mục lục sách. Không có
    embedding, không có chunk, nên giữ được quan hệ thứ bậc giữa các phần.

    Vì vậy đây là fallback hợp lý khi hybrid search cho điểm cosine thấp: điểm
    thấp thường có nghĩa là câu hỏi không ăn khớp với bất kỳ đoạn rời rạc nào.

⚠ Lưu ý về API (đã kiểm chứng, đừng đoán schema từ ví dụ code cũ):
    - Endpoint /retrieval đã deprecated (vẫn chạy, response có field "deprecation")
    - Kết quả nằm trong "retrieved_nodes", mỗi node có "relevant_contents" là
      list LỒNG list: list[list[{section_title, relevant_content}]]
    - API KHÔNG trả điểm số -> phải tự gán điểm theo thứ hạng

Chạy:
    python -m src.task8_pageindex_vectorless
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
PDF_CACHE_DIR = Path(__file__).parent.parent / "pageindex_pdfs"
DOC_IDS_FILE = Path(__file__).parent.parent / "pageindex_doc_ids.json"

# Chỉ upload văn bản luật. Bài tư vấn ngắn và phẳng, không có cấu trúc chương/mục
# nên PageIndex chẳng khai thác được gì thêm so với hybrid search — mà mỗi tài
# liệu upload đều tốn quota và thời gian dựng cây.
UPLOAD_ONLY_LEGAL = True

POLL_INTERVAL_SEC = 3
POLL_TIMEOUT_SEC = 120


# =============================================================================
# Chuẩn bị tài liệu
# =============================================================================

def _markdown_to_pdf(md_path: Path, pdf_path: Path) -> bool:
    """
    Chuyển .md sang PDF vì PageIndex chỉ nhận PDF, không nhận markdown.

    Phải nhúng font Unicode: font lõi của fpdf2 chỉ hỗ trợ latin-1, gặp tiếng
    Việt có dấu là ném UnicodeEncodeError. Dùng font hệ thống của Windows.
    """
    from fpdf import FPDF

    font_candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    font_path = next((f for f in font_candidates if f.exists()), None)
    if font_path is None:
        print(f"  ✗ Không tìm thấy font Unicode để dựng PDF tiếng Việt")
        return False

    pdf = FPDF()
    pdf.add_font("uni", "", str(font_path))
    pdf.set_font("uni", size=10)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    for line in md_path.read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if not line:
            pdf.ln(3)
            continue
        try:
            pdf.multi_cell(0, 5, line)
        except Exception:
            continue  # bỏ qua dòng dựng lỗi, không làm hỏng cả file

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(pdf_path))
    return True


def _get_client():
    """Khởi tạo PageIndex client, trả None nếu chưa sẵn sàng."""
    if not PAGEINDEX_API_KEY:
        return None
    try:
        from pageindex import PageIndexClient
    except ImportError:
        try:
            from pageindex.client import PageIndexClient
        except ImportError:
            print("  ⚠ Chưa cài SDK: pip install pageindex")
            return None
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


# =============================================================================
# Upload
# =============================================================================

def upload_documents() -> dict[str, str]:
    """
    Upload tài liệu lên PageIndex và lưu lại doc_id.

    doc_id được cache ra file để lần chạy sau không upload lại — mỗi lần upload
    PageIndex phải dựng lại cây mục lục, vừa mất vài phút vừa tốn quota.

    Returns:
        dict {tên file: doc_id}
    """
    client = _get_client()
    if client is None:
        print("  ⚠ Chưa có PAGEINDEX_API_KEY trong .env — bỏ qua upload")
        return {}

    doc_ids: dict[str, str] = {}
    if DOC_IDS_FILE.exists():
        doc_ids = json.loads(DOC_IDS_FILE.read_text(encoding="utf-8"))

    search_dir = STANDARDIZED_DIR / "legal" if UPLOAD_ONLY_LEGAL else STANDARDIZED_DIR
    if not search_dir.exists():
        print(f"  ⚠ Chưa có {search_dir} — chạy Task 3 trước")
        return doc_ids

    for md_file in sorted(search_dir.rglob("*.md")):
        if md_file.name in doc_ids:
            print(f"  → Đã upload trước đó: {md_file.name}")
            continue

        pdf_path = PDF_CACHE_DIR / f"{md_file.stem}.pdf"
        if not pdf_path.exists() and not _markdown_to_pdf(md_file, pdf_path):
            continue

        try:
            resp = client.submit_document(str(pdf_path))
            doc_id = resp.get("doc_id") or resp.get("id")
            if doc_id:
                doc_ids[md_file.name] = doc_id
                print(f"  ✓ {md_file.name} → {doc_id}")
        except Exception as e:
            print(f"  ✗ Upload lỗi {md_file.name}: {type(e).__name__}: {e}")

    if doc_ids:
        DOC_IDS_FILE.write_text(
            json.dumps(doc_ids, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return doc_ids


# =============================================================================
# Truy vấn
# =============================================================================

def _parse_retrieval(retrieval: dict, top_k: int, doc_name: str) -> list[dict]:
    """
    Bóc kết quả từ response của PageIndex.

    Cấu trúc thật (đã kiểm chứng, khác với ví dụ trong tài liệu cũ):
        retrieved_nodes: [
            { relevant_contents: [ [ {section_title, relevant_content}, ... ] ] }
        ]
    relevant_contents là list LỒNG list nên phải duyệt 2 tầng.
    """
    results = []
    nodes = retrieval.get("retrieved_nodes") or []

    for node_rank, node in enumerate(nodes):
        for group in node.get("relevant_contents") or []:
            # Phòng trường hợp API trả về list phẳng thay vì list lồng
            items = group if isinstance(group, list) else [group]
            for item in items:
                if not isinstance(item, dict):
                    continue
                content = item.get("relevant_content") or ""
                if not content.strip():
                    continue

                # PageIndex không trả điểm -> tự gán giảm dần theo thứ hạng để
                # kết quả vẫn sắp xếp được và ghép được với các retriever khác.
                # Đây là điểm QUY ƯỚC, không phải độ tương đồng đo được — không
                # đem so với ngưỡng cosine của Task 9.
                score = round(1.0 / (1 + len(results)), 4)

                results.append({
                    "content": content,
                    "score": score,
                    "metadata": {
                        "source": doc_name,
                        "section": item.get("section_title", ""),
                        "type": "legal",
                        "node_rank": node_rank,
                    },
                    "source": "pageindex",
                    "retriever": "pageindex",
                })

                if len(results) >= top_k:
                    return results

    return results


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval bằng PageIndex. Dùng làm fallback ở Task 9.

    Args:
        query: Câu truy vấn
        top_k: Số kết quả tối đa

    Returns:
        List of {'content', 'score', 'metadata', 'source': 'pageindex'}.
        Trả LIST RỖNG (không ném lỗi) khi chưa cấu hình API key hoặc chưa upload
        tài liệu — fallback không dùng được thì pipeline chính vẫn phải chạy
        tiếp, không được kéo cả hệ thống sập theo.
    """
    client = _get_client()
    if client is None:
        return []

    if not DOC_IDS_FILE.exists():
        print("  ⚠ Chưa upload tài liệu lên PageIndex — chạy upload_documents()")
        return []

    doc_ids = json.loads(DOC_IDS_FILE.read_text(encoding="utf-8"))
    if not doc_ids:
        return []

    results: list[dict] = []

    for doc_name, doc_id in doc_ids.items():
        if len(results) >= top_k:
            break

        try:
            resp = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = resp.get("retrieval_id") or resp.get("id")
            if not retrieval_id:
                continue

            # Truy vấn chạy bất đồng bộ -> phải chờ tới khi completed
            deadline = time.time() + POLL_TIMEOUT_SEC
            retrieval = None
            while time.time() < deadline:
                retrieval = client.get_retrieval(retrieval_id)
                if retrieval.get("status") in ("completed", "failed"):
                    break
                time.sleep(POLL_INTERVAL_SEC)

            if not retrieval or retrieval.get("status") != "completed":
                continue

            results.extend(
                _parse_retrieval(retrieval, top_k - len(results), doc_name)
            )

        except Exception as e:
            print(f"  ⚠ PageIndex lỗi trên {doc_name}: {type(e).__name__}: {e}")
            continue

    return results[:top_k]


if __name__ == "__main__":
    print("=" * 66)
    print("Task 8: PageIndex Vectorless RAG")
    print("=" * 66)

    if not PAGEINDEX_API_KEY:
        print("\n⚠ Chưa có PAGEINDEX_API_KEY trong .env")
        print("  Đăng ký tại https://pageindex.ai/ rồi thêm vào .env:")
        print("      PAGEINDEX_API_KEY=pix_...")
        print("\n  pageindex_search() vẫn trả về [] để Task 9 không bị gãy.")
        print(f"  Kiểm tra: pageindex_search('thử việc') = {pageindex_search('thử việc')}")
        raise SystemExit(0)

    print("\nĐang upload tài liệu...")
    ids = upload_documents()
    print(f"→ {len(ids)} tài liệu sẵn sàng")

    print("\nTruy vấn thử:")
    for q in ["thời gian thử việc tối đa là bao lâu",
              "trình tự xử lý kỷ luật sa thải gồm những bước nào"]:
        print(f"\n  {q!r}")
        for r in pageindex_search(q, top_k=3):
            print(f"    [{r['score']:.3f}] ({r['metadata']['section']}) "
                  f"{r['content'][:80]}...")
