"""
Task 3 — Chuẩn hoá toàn bộ dữ liệu thô về Markdown.

Dùng MarkItDown (Microsoft): https://github.com/microsoft/markitdown
    pip install "markitdown[pdf]"    # thiếu extra [pdf] sẽ lỗi MissingDependencyException

Mỗi file .md xuất ra đều mang một header nguồn (tiêu đề, URL, số hiệu văn bản).
Đây không phải trang trí: Task 10 yêu cầu câu trả lời có citation dạng [Nguồn, Năm],
mà sau khi Task 4 cắt nhỏ văn bản thì chunk không còn biết mình đến từ đâu. Ghi
nguồn ngay vào văn bản chuẩn hoá là cách rẻ nhất để thông tin đó sống sót qua
toàn bộ pipeline.

Chạy:
    python -m src.task3_convert_markdown
"""

import hashlib
import json
import re
from pathlib import Path

from src.task1_collect_legal_docs import FULLTEXT_DOCS, LEGAL_DOCS

# MarkItDown kéo theo magika -> onnxruntime (~200MB). Trong buổi lab 3 tiếng với
# mạng chậm, chờ tải xong có thể ngốn hết một checkpoint. pdfminer.six chính là
# engine mà markitdown[pdf] gọi bên dưới để bóc text PDF, nên đường dự phòng này
# cho ra kết quả tương đương với PDF text-based — chỉ nhẹ hơn rất nhiều.
try:
    from markitdown import MarkItDown
    _HAS_MARKITDOWN = True
except ImportError:
    MarkItDown = None
    _HAS_MARKITDOWN = False

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

# filename -> thông tin nguồn, để gắn header cho file legal
_LEGAL_META = {doc["filename"]: doc for doc in (*LEGAL_DOCS, *FULLTEXT_DOCS)}

# Định dạng đầu vào cho phần legal. .html là bản toàn văn dạng text (xem Task 1);
# .pdf công báo là bản scan nên thường không bóc được chữ, giữ lại chủ yếu để
# chứng minh xuất xứ.
LEGAL_INPUT_SUFFIXES = (".html", ".htm", ".pdf", ".docx", ".doc")


def _clean_pdf_text(text: str) -> str:
    """
    Dọn nhiễu đặc trưng của PDF công báo.

    Các bản .signed.pdf có header/footer lặp lại ở mọi trang ("CÔNG BÁO/Số ...")
    và số trang đứng riêng một dòng. Nếu để nguyên, những dòng rác này lặp lại
    hàng trăm lần trong corpus và sẽ đội tần suất từ (term frequency) một cách
    giả tạo — BM25 ở Task 6 vì thế có thể xếp hạng nhầm chỉ vì chunk chứa nhiều
    chuỗi lặp vô nghĩa.
    """
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            cleaned.append("")
            continue
        # Số trang đứng một mình
        if re.fullmatch(r"\d{1,4}", stripped):
            continue
        # Header công báo lặp theo trang
        if re.match(r"^CÔNG BÁO\s*/\s*Số", stripped, re.IGNORECASE):
            continue
        if re.fullmatch(r"(?:CÔNG BÁO|Số\s*\d+\s*\+\s*\d+).{0,40}", stripped, re.IGNORECASE):
            continue

        cleaned.append(stripped)

    text = "\n".join(cleaned)
    # Gộp >2 dòng trống liên tiếp thành 1 dòng trống
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _legal_header(filename: str) -> str:
    """Sinh header nguồn cho một văn bản pháp luật."""
    meta = _LEGAL_META.get(filename)
    if not meta:
        return f"# {Path(filename).stem}\n\n"

    # Với bản toàn văn, ghi thêm URL văn bản chính thức trên cổng Chính phủ để
    # citation ở Task 10 trỏ về nguồn gốc chứ không dừng ở trang trung gian.
    official = meta.get("official_url")
    nguon = (
        f"**Nguồn chính thức:** {official}\n**Bản toàn văn:** {meta['url']}\n"
        if official else f"**Nguồn:** {meta['url']}\n"
    )

    return (
        f"# {meta['title']}\n\n"
        f"**Loại nguồn:** Văn bản quy phạm pháp luật\n"
        f"{nguon}"
        f"**Nội dung liên quan:** {meta['desc']}\n\n"
        f"---\n\n"
    )


# Widget của trang nguồn nằm xen giữa các điều khoản, không phải nội dung văn bản
_HTML_NOISE_SELECTORS = ["span.tooltip-button", ".target-hidden", ".bg-theo-doi"]
_HTML_NOISE_PHRASES = {"đang theo dõi", "theo dõi", "tải biểu mẫu", "tải về",
                       "lưu văn bản", "xem thêm", "thuộc tính", "tải word"}

_BLOCK_TAGS = ["p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "td", "th", "caption"]


def _extract_html_text(filepath: Path) -> str:
    """
    Chuyển HTML toàn văn sang text có cấu trúc.

    Hai thứ phải xử lý, nếu bỏ qua thì corpus hỏng nặng:

    1. Nút "Đang theo dõi" của trang nguồn chèn giữa mọi khoản — chiếm tới 1/3
       tổng số dòng. Để nguyên thì chunk bị rác ăn mất chỗ và IDF của BM25 bị
       méo vì mấy từ đó bỗng trở nên "phổ biến" khắp corpus.

    2. get_text("\\n") xuống dòng ở MỌI thẻ, kể cả <a> inline. Văn bản luật dẫn
       chiếu chằng chịt ("theo quy định của <a>Luật Doanh nghiệp</a>,") nên một
       câu bị vỡ thành nhiều mảnh, có dòng chỉ còn đúng dấu phẩy. Câu vỡ làm
       hỏng cả embedding (Task 5) lẫn ranh giới chunk (Task 4).
       Cách xử lý: làm phẳng từng khối (p/li/h*/td) thành một dòng trước, rồi
       mới xuống dòng theo khối.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(filepath.read_text(encoding="utf-8"), "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for selector in _HTML_NOISE_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()

    # Làm phẳng khối: gộp toàn bộ text con (kể cả thẻ inline) thành 1 dòng
    for el in soup.find_all(_BLOCK_TAGS):
        el.string = el.get_text(" ", strip=True)

    # Bảng -> mỗi hàng 1 dòng, các ô ngăn bằng " | " (giữ được ranh giới vùng
    # lương trong biểu mức lương tối thiểu)
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c]
            if cells:
                rows.append(" | ".join(cells))
        table.replace_with("\n" + "\n".join(rows) + "\n")

    lines = []
    for raw in soup.get_text("\n").split("\n"):
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            continue
        if line.lower() in _HTML_NOISE_PHRASES:
            continue
        # Dòng chỉ còn dấu câu — mảnh vụn sót lại, không mang thông tin
        if not re.search(r"\w", line):
            continue
        lines.append(line)

    return "\n".join(lines)


def _extract_text(filepath: Path, md_converter) -> str:
    """Bóc text thô từ 1 file theo định dạng."""
    suffix = filepath.suffix.lower()

    if suffix in (".html", ".htm"):
        return _extract_html_text(filepath)

    if md_converter is not None:
        return md_converter.convert(str(filepath)).text_content or ""

    if suffix != ".pdf":
        raise RuntimeError(
            f"Không có MarkItDown nên chỉ xử lý được PDF/HTML, gặp {suffix}"
        )

    from pdfminer.high_level import extract_text
    return extract_text(str(filepath)) or ""


def convert_legal_docs() -> int:
    """Convert PDF/DOCX trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not legal_dir.exists():
        print("  ⚠ Chưa có data/landing/legal/ — chạy Task 1 trước")
        return 0

    md = MarkItDown() if _HAS_MARKITDOWN else None
    print(f"  Engine: {'MarkItDown' if md else 'pdfminer.six (fallback)'}")
    count = 0

    # Khử trùng lặp theo nội dung file, không theo tên.
    # data/landing/legal/ hiện có vài văn bản bị lưu 2 lần dưới hai tên khác nhau
    # (45.signed.pdf và bo-luat-lao-dong-2019.pdf là cùng một file). Nếu convert
    # cả hai, mỗi điều luật sẽ nằm 2 lần trong corpus: chunk trùng tự cạnh tranh
    # trong RRF, chiếm chỗ của tài liệu khác trong top_k, và làm lệch IDF của BM25
    # (từ trong văn bản nhân đôi trông có vẻ "phổ biến" hơn thực tế).
    #
    # Ưu tiên giữ file có tên mô tả rõ (nằm trong LEGAL_DOCS) để còn gắn được
    # header nguồn phục vụ citation ở Task 10.
    files = [f for f in legal_dir.iterdir()
             if f.is_file() and f.suffix.lower() in LEGAL_INPUT_SUFFIXES]
    files.sort(key=lambda f: (0 if f.name in _LEGAL_META else 1, f.name))

    seen_hashes: dict[str, str] = {}

    for filepath in files:
        digest = hashlib.md5(filepath.read_bytes()).hexdigest()
        if digest in seen_hashes:
            print(f"  ⊘ Bỏ qua bản trùng: {filepath.name} "
                  f"(giống hệt {seen_hashes[digest]})")
            continue
        seen_hashes[digest] = filepath.name

        print(f"  Converting: {filepath.name}")
        try:
            raw_text = _extract_text(filepath, md)
        except Exception as e:
            print(f"    ✗ Lỗi: {type(e).__name__}: {e}")
            continue

        body = _clean_pdf_text(raw_text)
        if len(body) < 200:
            print(f"    ⊘ Chỉ trích được {len(body)} ký tự — bản scan ảnh, "
                  f"không có lớp text. Dùng bản *-toanvan.html thay thế.")
            continue

        content = _legal_header(filepath.name) + body
        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(content, encoding="utf-8")
        count += 1
        print(f"    ✓ {output_path.name} — {len(content):,} ký tự")

    return count


def convert_news_articles() -> int:
    """Convert JSON bài viết trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not news_dir.exists():
        print("  ⚠ Chưa có data/landing/news/ — chạy Task 2 trước")
        return 0

    count = 0

    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() != ".json":
            continue

        print(f"  Converting: {filepath.name}")
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"    ✗ JSON hỏng: {e}")
            continue

        body = (data.get("content_markdown") or "").strip()
        if len(body) < 200:
            print(f"    ✗ Nội dung quá ngắn ({len(body)} ký tự), bỏ qua")
            continue

        # Bỏ dòng '# tiêu đề' đã có sẵn trong body để không lặp với header
        body = re.sub(r"^#\s+.*\n+", "", body, count=1)

        header = (
            f"# {data.get('title', 'Unknown')}\n\n"
            f"**Loại nguồn:** Bài viết hướng dẫn / tư vấn pháp luật\n"
            f"**Nguồn:** {data.get('url', 'N/A')}\n"
            f"**Ngày thu thập:** {data.get('date_crawled', 'N/A')}\n\n"
            f"---\n\n"
        )

        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(header + body, encoding="utf-8")
        count += 1
        print(f"    ✓ {output_path.name} — {len(header) + len(body):,} ký tự")

    return count


def convert_all() -> None:
    """Convert toàn bộ dữ liệu thô."""
    print("=" * 60)
    print("Task 3: Chuẩn hoá về Markdown (MarkItDown)")
    print("=" * 60)

    print("\n--- Văn bản pháp luật ---")
    n_legal = convert_legal_docs()

    print("\n--- Bài viết tư vấn ---")
    n_news = convert_news_articles()

    print()
    print("-" * 60)
    print(f"Kết quả: {n_legal} văn bản luật + {n_news} bài viết = "
          f"{n_legal + n_news} file .md")
    print(f"Output: {OUTPUT_DIR}")

    if n_legal and n_news:
        print("✓ Đạt yêu cầu Task 3 (cả legal/ và news/ đều có .md)")
    else:
        print("⚠ Cần có .md ở cả legal/ và news/")


if __name__ == "__main__":
    convert_all()
