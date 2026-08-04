"""
Task 2 — Crawl bài viết tư vấn / giải thích pháp luật lao động.

Vì sao cần lớp dữ liệu này bên cạnh văn bản luật gốc (Task 1):
    Văn bản luật viết theo ngôn ngữ lập pháp ("người lao động", "người sử dụng
    lao động", "đơn phương chấm dứt hợp đồng"), còn người dùng Gen Z hỏi bằng
    ngôn ngữ đời thường ("bị đuổi việc qua Zalo", "tăng ca không lương").
    Các bài tư vấn là cầu nối từ vựng giữa hai bên — thiếu chúng thì semantic
    search rất dễ trượt vì query và văn bản luật gần như không chung từ nào.

Nguồn: baochinhphu.vn (Báo Điện tử Chính phủ) và laodong.vn (Báo Lao Động,
chuyên mục Tư vấn pháp luật) — đều là trang công khai, cho phép đọc tự do.

Chạy:
    python -m src.task2_crawl_news
"""

import asyncio
import json
import re
import time
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9",
}

# 10 bài cho yêu cầu tối thiểu 5 — dư biên phòng khi vài link đổi/chết.
# Chủ đề bám sát 2 câu hỏi mẫu của nhóm: thử việc, OT, nghỉ phép, sa thải.
#
# Đã bỏ laodong.vn: site trả về một trang JS đặt cookie rồi reload
# (`document.cookie="D1N=..."; window.location.reload()`), requests thường chỉ
# nhận được 177 byte rỗng. Theo đúng hướng dẫn trong README của lab — gặp trang
# chặn bot thì đổi nguồn khác thay vì tìm cách vượt qua.
ARTICLE_URLS: list[str] = [
    # --- Thử việc ---
    "https://baochinhphu.vn/thoi-gian-thu-viec-duoc-tinh-ngay-nghi-phep-10296422.htm",
    # --- Làm thêm giờ (OT) ---
    "https://baochinhphu.vn/lam-them-gio-the-nao-la-dung-quy-dinh-102240130085419703.htm",
    "https://baochinhphu.vn/can-dieu-kien-gi-de-su-dung-nguoi-lao-dong-lam-them-gio-102291610.htm",
    # --- Nghỉ phép năm ---
    "https://baochinhphu.vn/hieu-the-nao-ve-quy-dinh-tinh-gop-ngay-phep-nam-102240613101442693.htm",
    "https://baochinhphu.vn/khong-nghi-het-phep-nam-co-duoc-thanh-toan-tien-10225090516273483.htm",
    # --- Sa thải / kỷ luật lao động ---
    "https://baochinhphu.vn/khi-nao-thi-bi-xu-ly-ky-luat-sa-thai-102215444.htm",
    "https://baochinhphu.vn/dieu-kien-ap-dung-hinh-thuc-sa-thai-lao-dong-102251208144356042.htm",
    "https://baochinhphu.vn/xu-ly-the-nao-khi-nguoi-lao-dong-nghi-viec-5-ngay-lien-tuc-tro-len-102241003162949239.htm",
    # --- Chấm dứt hợp đồng trái luật (câu hỏi mẫu: bị đuổi việc không báo trước) ---
    "https://baochinhphu.vn/the-nao-la-don-phuong-cham-dut-hop-dong-lao-dong-dung-luat-102230428144412753.htm",
    "https://baochinhphu.vn/truong-hop-nao-bi-coi-la-cham-dut-hop-dong-lao-dong-trai-luat-102220905124022519.htm",
]

MIN_CONTENT_CHARS = 500  # test_news_files_have_content yêu cầu file > 500 bytes


# =============================================================================
# Đường 1: Crawl4AI (thư viện khuyến nghị của bài lab)
# =============================================================================

def _crawl4ai_available() -> bool:
    """Kiểm tra crawl4ai + playwright browser đã sẵn sàng chưa."""
    try:
        import crawl4ai  # noqa: F401
        return True
    except ImportError:
        return False


async def crawl_article_crawl4ai(url: str) -> dict:
    """Crawl 1 bài bằng Crawl4AI (headless browser, render được JS)."""
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler(verbose=False) as crawler:
        result = await crawler.arun(url=url)
        markdown = getattr(result, "markdown", "") or ""
        # crawl4ai >=0.4 trả object MarkdownGenerationResult thay vì str
        if not isinstance(markdown, str):
            markdown = getattr(markdown, "raw_markdown", "") or str(markdown)

        metadata = getattr(result, "metadata", None) or {}
        return {
            "url": url,
            "title": metadata.get("title") or "Unknown",
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": markdown.strip(),
            "crawler": "crawl4ai",
        }


# =============================================================================
# Đường 2: requests + BeautifulSoup (fallback, không cần browser binary)
# =============================================================================

def _clean_soup(soup):
    """Bỏ các thẻ không phải nội dung bài viết."""
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "form", "iframe", "noscript"]):
        tag.decompose()
    return soup


def _extract_main_content(soup) -> str:
    """
    Trích phần thân bài.

    Thử selector riêng của từng báo trước; nếu trang đổi layout thì rơi về
    heuristic "khối chứa nhiều text trong thẻ <p> nhất" — cách này không phụ
    thuộc vào class name nên sống sót qua các lần site redesign.
    """
    known_selectors = [
        "div.detail-content", "div#abody", "div.article-body",   # baochinhphu
        "div.art-body", "div.article-content",                    # laodong
        "article",
    ]

    for selector in known_selectors:
        node = soup.select_one(selector)
        if node:
            text = "\n\n".join(
                p.get_text(" ", strip=True)
                for p in node.find_all("p")
                if len(p.get_text(strip=True)) > 20
            )
            if len(text) >= MIN_CONTENT_CHARS:
                return text

    # Heuristic: chọn container có tổng text trong <p> lớn nhất
    best_text = ""
    for div in soup.find_all(["div", "section", "article"]):
        paragraphs = div.find_all("p", recursive=False) or div.find_all("p")
        text = "\n\n".join(
            p.get_text(" ", strip=True)
            for p in paragraphs
            if len(p.get_text(strip=True)) > 20
        )
        if len(text) > len(best_text):
            best_text = text

    return best_text


def crawl_article_requests(url: str) -> dict:
    """Crawl 1 bài bằng requests + BeautifulSoup."""
    import requests
    from bs4 import BeautifulSoup

    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"

    soup = _clean_soup(BeautifulSoup(resp.text, "lxml"))

    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text(" ", strip=True) if title_tag else "Unknown"

    body = _extract_main_content(soup)
    content_markdown = f"# {title}\n\n{body}" if body else ""

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content_markdown,
        "crawler": "requests+bs4",
    }


# =============================================================================
# Điều phối
# =============================================================================

def _slugify(url: str) -> str:
    """Sinh tên file từ URL — dễ truy nguồn hơn article_01.json."""
    slug = url.rstrip("/").split("/")[-1]
    slug = re.sub(r"\.(htm|html|ldo)$", "", slug)
    slug = re.sub(r"-\d{6,}$", "", slug)          # bỏ ID bài dài ở cuối
    slug = re.sub(r"[^a-zA-Z0-9\-]", "-", slug)
    return re.sub(r"-+", "-", slug).strip("-")[:80] or "article"


def setup_directory() -> None:
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


async def crawl_all() -> int:
    """Crawl toàn bộ ARTICLE_URLS. Trả về số bài lưu thành công."""
    print("=" * 60)
    print("Task 2: Crawl bài viết tư vấn pháp luật lao động")
    print("=" * 60)

    setup_directory()

    use_crawl4ai = _crawl4ai_available()
    print(f"Crawler: {'Crawl4AI' if use_crawl4ai else 'requests + BeautifulSoup'}")
    print()

    success = 0
    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] {url}")
        try:
            if use_crawl4ai:
                article = await crawl_article_crawl4ai(url)
                # Browser có thể trả trang rỗng khi bị chặn -> thử lại bằng requests
                if len(article["content_markdown"]) < MIN_CONTENT_CHARS:
                    print("  … Crawl4AI trả nội dung quá ngắn, thử lại bằng requests")
                    article = crawl_article_requests(url)
            else:
                article = crawl_article_requests(url)
        except Exception as e:
            print(f"  ✗ Lỗi: {type(e).__name__}: {e}")
            continue

        if len(article["content_markdown"]) < MIN_CONTENT_CHARS:
            print(f"  ✗ Nội dung quá ngắn ({len(article['content_markdown'])} chars), bỏ qua")
            continue

        filepath = DATA_DIR / f"{_slugify(url)}.json"
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        success += 1
        print(f"  ✓ {filepath.name} — {len(article['content_markdown'])} chars")
        print(f"    \"{article['title'][:70]}\"")

        time.sleep(1)  # tránh dồn request vào server báo

    print()
    print("-" * 60)
    print(f"Kết quả: {success}/{len(ARTICLE_URLS)} bài")
    if success >= 5:
        print("✓ Đạt yêu cầu Task 2 (tối thiểu 5 bài)")
    else:
        print("⚠ Chưa đủ 5 bài — bổ sung URL vào ARTICLE_URLS rồi chạy lại")

    return success


if __name__ == "__main__":
    asyncio.run(crawl_all())
