"""Day 8 — RAG Pipeline v2: Trợ lý hỏi đáp Luật Lao động cho người trẻ."""

import sys

# Console Windows mặc định dùng codepage cp1252 -> mọi lệnh print() có dấu tiếng
# Việt đều ném UnicodeEncodeError trước khi script kịp làm gì. Toàn bộ dữ liệu của
# nhóm là tiếng Việt nên bật UTF-8 ngay tại package, không bắt từng thành viên
# phải tự set biến môi trường PYTHONIOENCODING.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass
