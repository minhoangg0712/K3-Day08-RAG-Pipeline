"""
Task 1 — Thu thập văn bản quy phạm pháp luật về lao động.

Chủ đề nhóm: "Trợ Lý Hỏi Đáp Luật Lao Động Cho Người Trẻ" — tra cứu các vấn đề
pháp lý lao động phổ biến với Gen Z (thử việc, OT, nghỉ phép, hợp đồng, sa thải).

Nguồn dữ liệu: chỉ dùng bản PDF *có chữ ký số* từ cổng thông tin chính thức của
Chính phủ (datafiles.chinhphu.vn). Đây là bản gốc do cơ quan ban hành phát hành,
không phải bản đánh máy lại của các trang tổng hợp — quan trọng với hệ thống hỏi
đáp pháp luật vì sai một chữ trong điều khoản là sai cả câu trả lời.

Chạy:
    python -m src.task1_collect_legal_docs
"""

import time
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

# URL file nằm trên datafiles.chinhphu.vn nhưng KHÔNG theo một quy tắc đặt tên cố
# định (45.signed.pdf, 12-2022-nd.signed.pdf, 293-cp.signed.pdf, tháng có/không có
# số 0 ở đầu...). Đừng suy ra URL từ số hiệu — mở trang văn bản trên
# https://vanban.chinhphu.vn rồi copy đúng link đính kèm.
#
# Chỉ đưa vào corpus văn bản ĐANG CÓ HIỆU LỰC tại thời điểm làm bài (04/08/2026).
# Đây là quyết định về chất lượng dữ liệu, không phải chi tiết vụn vặt: RAG không
# tự biết văn bản nào đã bị thay thế, nó chỉ so khớp ngữ nghĩa. Trộn bản cũ và bản
# mới vào cùng một vector store thì retrieval sẽ trả về cả hai mức phạt / hai mức
# lương khác nhau, và LLM sẽ trích dẫn nhầm một cách rất thuyết phục.
#   - Bỏ NĐ 74/2024 (lương tối thiểu): đã bị NĐ 293/2025 thay thế từ 01/01/2026.
#   - Bỏ NĐ 283/2026 (xử phạt): ban hành 15/07/2026 nhưng tới 10/09/2026 mới có
#     hiệu lực, hiện NĐ 12/2022 vẫn là văn bản áp dụng.
LEGAL_DOCS: list[dict] = [
    {
        "filename": "bo-luat-lao-dong-2019.pdf",
        "url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2019/12/45.signed.pdf",
        "title": "Bộ luật Lao động số 45/2019/QH14",
        "desc": "Văn bản gốc: thử việc (Điều 24-27), thời giờ làm việc & làm thêm "
                "(Điều 105-107), nghỉ phép năm (Điều 113), kỷ luật sa thải (Điều 124-127)",
    },
    {
        "filename": "nghi-dinh-145-2020-huong-dan-blld.pdf",
        "url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2020/12/145.signed.pdf",
        "title": "Nghị định 145/2020/NĐ-CP",
        "desc": "Hướng dẫn thi hành BLLĐ về điều kiện lao động và quan hệ lao động - "
                "chi tiết cách tính phép năm, giới hạn giờ làm thêm, trình tự xử lý kỷ luật",
    },
    {
        "filename": "nghi-dinh-12-2022-xu-phat-vphc-lao-dong.pdf",
        "url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2022/01/12-2022-nd.signed.pdf",
        "title": "Nghị định 12/2022/NĐ-CP (hiệu lực 17/01/2022)",
        "desc": "Xử phạt vi phạm hành chính lĩnh vực lao động, BHXH - dùng để trả lời "
                "'công ty làm vậy có bị phạt không, phạt bao nhiêu'",
    },
    {
        "filename": "nghi-dinh-293-2025-luong-toi-thieu.pdf",
        "url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2025/11/293-cp.signed.pdf",
        "title": "Nghị định 293/2025/NĐ-CP (hiệu lực 01/01/2026)",
        "desc": "Mức lương tối thiểu vùng hiện hành - căn cứ tính lương thử việc "
                "tối thiểu (85% lương chính thức, Điều 26 BLLĐ)",
    },
]

HEADERS = {
    # Cổng chinhphu.vn trả 403 với User-Agent mặc định của requests
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}

MIN_SIZE_BYTES = 1024  # test_files_not_empty yêu cầu > 1KB


def setup_directory() -> None:
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


def download_file(doc: dict, timeout: int = 60) -> bool:
    """
    Tải một văn bản về DATA_DIR.

    Kiểm tra magic bytes '%PDF' thay vì chỉ dựa vào HTTP 200: khi bị WAF chặn,
    server vẫn trả 200 kèm một trang HTML "Access Denied" — nếu chỉ check status
    code thì file rác đó sẽ lọt vào corpus và làm hỏng kết quả retrieval.

    Returns:
        True nếu tải và xác thực thành công.
    """
    filepath = DATA_DIR / doc["filename"]

    if filepath.exists() and filepath.stat().st_size > MIN_SIZE_BYTES:
        print(f"  → Đã có sẵn, bỏ qua: {doc['filename']} "
              f"({filepath.stat().st_size / 1024:.0f} KB)")
        return True

    try:
        resp = requests.get(doc["url"], headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  ✗ Lỗi tải {doc['filename']}: {e}")
        return False

    content = resp.content

    if not content.startswith(b"%PDF"):
        print(f"  ✗ {doc['filename']}: nội dung không phải PDF "
              f"(có thể bị WAF chặn) - {len(content)} bytes")
        return False

    if len(content) <= MIN_SIZE_BYTES:
        print(f"  ✗ {doc['filename']}: file quá nhỏ ({len(content)} bytes)")
        return False

    filepath.write_bytes(content)
    print(f"  ✓ {doc['filename']} ({len(content) / 1024:.0f} KB) — {doc['title']}")
    return True


def collect_all() -> int:
    """Tải toàn bộ văn bản trong LEGAL_DOCS. Trả về số file thành công."""
    print("=" * 60)
    print("Task 1: Thu thập văn bản pháp luật lao động")
    print("=" * 60)

    setup_directory()
    print()

    success = 0
    for i, doc in enumerate(LEGAL_DOCS, 1):
        print(f"[{i}/{len(LEGAL_DOCS)}] {doc['title']}")
        if download_file(doc):
            success += 1
        time.sleep(1)  # lịch sự với server công

    print()
    print("-" * 60)
    print(f"Kết quả: {success}/{len(LEGAL_DOCS)} văn bản")

    if success >= 3:
        print("✓ Đạt yêu cầu Task 1 (tối thiểu 3 văn bản)")
    else:
        print(f"⚠ Chưa đủ 3 văn bản. Tải thủ công từ https://vanban.chinhphu.vn "
              f"và lưu vào {DATA_DIR}")

    return success


if __name__ == "__main__":
    collect_all()
