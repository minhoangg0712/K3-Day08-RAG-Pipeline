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
import re
import time
import unicodedata
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


# =============================================================================
# Vectorless chạy local — dùng khi chưa có PAGEINDEX_API_KEY
# =============================================================================

# Tiêu đề điều luật: "Điều 25. Thời gian thử việc"
_ARTICLE_RE = re.compile(r"^Điều\s+(\d+)\s*\.\s*(.*)$", re.MULTILINE)
# Tiêu đề chương: "Chương III" / "Chương III HỢP ĐỒNG LAO ĐỘNG"
_CHAPTER_RE = re.compile(r"^Chương\s+([IVXLC]+)\s*(.*)$", re.MULTILINE)

_LOCAL_STOPWORDS = {
    "là", "của", "và", "có", "cho", "được", "thì", "khi", "không", "phải",
    "tôi", "bao", "nhiêu", "gì", "nào", "với", "trong", "một", "các", "này",
    "mà", "để", "ở", "về", "hay", "bằng", "sau", "trước", "đã", "sẽ", "còn",
    "theo", "những", "người", "lao", "động",
}


def _tokens(text: str) -> set[str]:
    ws = re.findall(r"[0-9a-zà-ỹ]+", unicodedata.normalize("NFC", text).lower())
    return {w for w in ws if len(w) > 1 and w not in _LOCAL_STOPWORDS}


def _build_article_tree() -> list[dict]:
    """
    Dựng cây Chương → Điều từ các file markdown văn bản luật.

    Đây là bản địa phương hoá ý tưởng của PageIndex: thay vì cắt tài liệu thành
    chunk cố định rồi so vector, ta giữ nguyên ranh giới ĐIỀU LUẬT — đơn vị ngữ
    nghĩa thật sự của văn bản quy phạm — và ghi lại chương cha của nó. Một điều
    luật vì thế luôn về nguyên vẹn, không bị cắt ngang giữa các khoản.
    """
    tree: list[dict] = []

    for md_file in sorted((STANDARDIZED_DIR / "legal").glob("*.md")):
        text = md_file.read_text(encoding="utf-8")

        chapters = [(m.start(), f"Chương {m.group(1)} {m.group(2)}".strip())
                    for m in _CHAPTER_RE.finditer(text)]

        matches = list(_ARTICLE_RE.finditer(text))
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            # Chương gần nhất phía trên điều này
            chapter = ""
            for pos, name in chapters:
                if pos < start:
                    chapter = name
                else:
                    break

            tree.append({
                "article_no": m.group(1),
                "title": f"Điều {m.group(1)}. {m.group(2)}".strip(),
                "chapter": chapter,
                "body": text[start:end].strip(),
                "doc": md_file.name,
            })

    return tree


_TREE_CACHE: list[dict] | None = None


def _local_structural_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Duyệt cây điều luật bằng từ khoá — không embedding, không chunk cố định.

    Cách chấm điểm bắt chước hành vi tra mục lục: TIÊU ĐỀ điều luật được tính
    trọng số gấp 3 lần thân bài. Người tra cứu văn bản luật cũng làm đúng vậy —
    đọc lướt tên điều trước, thấy khớp mới mở ra đọc nội dung.

    Ưu tiên tuyệt đối cho truy vấn gọi thẳng số hiệu ("Điều 25 quy định gì"):
    khớp đúng số điều thì đẩy lên đầu, vì đó là ý định tra cứu rõ ràng nhất.
    """
    global _TREE_CACHE
    if _TREE_CACHE is None:
        _TREE_CACHE = _build_article_tree()
    if not _TREE_CACHE:
        return []

    q_tokens = _tokens(query)
    if not q_tokens:
        return []

    asked_articles = set(re.findall(r"[Đđ]iều\s+(\d+)", query))

    scored = []
    for node in _TREE_CACHE:
        title_hits = len(q_tokens & _tokens(node["title"]))
        body_hits = len(q_tokens & _tokens(node["body"]))
        chapter_hits = len(q_tokens & _tokens(node["chapter"]))

        score = 3.0 * title_hits + 1.0 * body_hits + 1.5 * chapter_hits
        if node["article_no"] in asked_articles:
            score += 100.0

        if score > 0:
            scored.append((score, node))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]
    max_score = top[0][0] or 1.0

    return [
        {
            "content": node["body"][:2000],
            # Chuẩn hoá về [0,1] để Task 9 và UI hiển thị cùng thang với cosine
            "score": round(score / max_score, 4),
            "metadata": {
                "source": node["doc"],
                "type": "legal",
                "section": node["title"],
                "chapter": node["chapter"],
                "engine": "local-structural",
            },
            "source": "pageindex",
        }
        for score, node in top
    ]


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval. Dùng làm fallback ở Task 9.

    Hai đường chạy:
        1. PageIndex hosted API — khi có PAGEINDEX_API_KEY và đã upload tài liệu.
        2. Duyệt cây điều luật CHẠY LOCAL — khi chưa cấu hình API key.

    Đường 2 không phải PageIndex thật, và không giả vờ là như vậy: kết quả gắn
    `metadata.engine = "local-structural"` để phân biệt. Nó tồn tại vì fallback
    mà trả rỗng thì coi như không có fallback — Task 9 sẽ buộc phải trả về kết
    quả hybrid điểm thấp cho đúng những truy vấn mà hybrid đã bó tay.

    Trường `source` vẫn là `"pageindex"` ở cả hai đường vì đó là tên GIAI ĐOẠN
    trong pipeline (nhánh vectorless), không phải tên nhà cung cấp.

    Args:
        query: Câu truy vấn
        top_k: Số kết quả tối đa

    Returns:
        List of {'content', 'score', 'metadata', 'source': 'pageindex'}.
        Không bao giờ ném lỗi — fallback hỏng thì pipeline chính vẫn phải chạy.
    """
    client = _get_client()
    if client is None:
        return _local_structural_search(query, top_k)

    if not DOC_IDS_FILE.exists():
        print("  ⚠ Chưa upload tài liệu lên PageIndex — dùng cây điều luật local")
        return _local_structural_search(query, top_k)

    doc_ids = json.loads(DOC_IDS_FILE.read_text(encoding="utf-8"))
    if not doc_ids:
        return _local_structural_search(query, top_k)

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
        print("\n→ Đang chạy đường 2: duyệt cây điều luật LOCAL (không phải PageIndex thật)")

        tree = _build_article_tree()
        print(f"  Cây điều luật: {len(tree)} điều từ "
              f"{len(set(n['doc'] for n in tree))} văn bản")

        for q in ["thời gian thử việc tối đa là bao lâu",
                  "trình tự xử lý kỷ luật sa thải gồm những bước nào",
                  "Điều 98 quy định gì"]:
            print(f"\n  {q!r}")
            for r in pageindex_search(q, top_k=3):
                print(f"    [{r['score']:.3f}] ({r['metadata']['section']}) "
                      f"{' '.join(r['content'].split())[:70]}...")
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
