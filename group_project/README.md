# Bài Tập Nhóm — Trợ Lý Hỏi Đáp Luật Lao Động Cho Người Trẻ

Trợ lý AI tra cứu và giải đáp các vấn đề pháp lý lao động phổ biến với Gen Z:
thử việc, làm thêm giờ (OT), nghỉ phép, hợp đồng, sa thải.

---

## 1. Nguồn Dữ Liệu

| Loại | Tài liệu | Số file |
|---|---|---|
| Văn bản luật (`data/landing/legal/`) | Bộ luật Lao động 2019; NĐ 145/2020 hướng dẫn BLLĐ; NĐ 12/2022 xử phạt VPHC lĩnh vực lao động; NĐ 293/2025 lương tối thiểu vùng | 4 |
| Bài tư vấn (`data/landing/news/`) | Bài hướng dẫn pháp luật lao động crawl từ nguồn công khai (thử việc, OT, phép năm, sa thải, chấm dứt HĐLĐ) | 10 |

Sau Task 3 → `data/standardized/` (Markdown) → Task 4 → **1069 chunk** trong ChromaDB.

> **Lưu ý về file PDF gốc:** bản PDF công báo của các văn bản luật là bản scan có
> chữ ký số, MarkItDown trích được rất ít text. Nhóm bổ sung thêm bản toàn văn
> dạng HTML (`*-toanvan.html`) và convert từ đó, nên `data/standardized/legal/`
> mới có nội dung đầy đủ để index.

---

## 2. Kiến Trúc Hệ Thống

```
                         ┌──────────────────────┐
   data/landing/  ──────►│ Task 3: MarkItDown   │──► data/standardized/*.md
   (PDF, HTML, JSON)     └──────────────────────┘             │
                                                              ▼
                                          ┌───────────────────────────────────┐
                                          │ Task 4: RecursiveCharacterSplitter│
                                          │   chunk 800 / overlap 100         │
                                          │   embed BAAI/bge-m3 (1024-d)      │
                                          │   → ChromaDB (1069 chunk)         │
                                          └───────────────────────────────────┘
                                                              │
   ┌──────────────────────────────────────────────────────────┴────────────────┐
   │                                                                           │
   ▼                                                                           ▼
┌──────────────────────────┐                              ┌────────────────────────────┐
│ Task 5: semantic_search  │                              │ Task 6: lexical_search     │
│  cosine trên ChromaDB    │                              │  BM25Okapi (+ TF-IDF)      │
│  (+ HyDE tuỳ chọn)       │                              │  tokenizer giữ số điều luật│
└────────────┬─────────────┘                              └─────────────┬──────────────┘
             │            top_k × 4 ứng viên mỗi nhánh                  │
             └──────────────────────┬─────────────────────────────────-─┘
                                    ▼
                    ┌─────────────────────────────────┐
                    │ Task 7: rerank_rrf(k=60)        │
                    │   RRF(d) = Σ 1/(60 + rank_r(d)) │
                    └────────────────┬────────────────┘
                                     ▼
                    ┌──────────────────────────────────────────────┐
                    │ Task 9: retrieve()                           │
                    │  cosine GỐC tốt nhất ≥ 0.48 ?                │
                    │    ├─ có  → trả kết quả hybrid               │
                    │    └─ không → Task 8: PageIndex (vectorless) │
                    └────────────────┬─────────────────────────────┘
                                     ▼
                    ┌──────────────────────────────────────────────┐
                    │ Task 10: generate_with_citation()            │
                    │  reorder [1,3,5,4,2] chống lost-in-the-middle│
                    │  → LLM (OpenRouter) → answer + [Nguồn, Điều] │
                    └────────────────┬─────────────────────────────┘
                                     ▼
                    ┌──────────────────────────────────────────────┐
                    │ app.py — Streamlit chat UI                   │
                    │  top_k / HyDE / ngưỡng fallback, nguồn,      │
                    │  chẩn đoán retrieval, conversation memory    │
                    └──────────────────────────────────────────────┘
```

### Điểm kỹ thuật đáng lưu ý

**Ngưỡng fallback so với cosine gốc, không phải điểm RRF.**
Điểm RRF sau khi fuse chỉ phụ thuộc *thứ hạng*: tài liệu đứng đầu luôn được đúng
`1/(60+1) ≈ 0.0164` bất kể nội dung có liên quan hay không. Nếu đem con số đó so
với ngưỡng `0.48` thì điều kiện luôn đúng → fallback chạy ở **mọi** truy vấn; còn
nếu hạ ngưỡng xuống dưới `0.0164` cho "hợp thang đo" thì fallback **không bao giờ**
chạy, kể cả với câu hỏi hoàn toàn lạc đề. Cả hai hướng đều sai mà test vẫn xanh.
Vì vậy `task9_retrieval_pipeline.py` so ngưỡng với **cosine similarity gốc** từ
`semantic_search` — đại lượng nằm trong `[0,1]` và thực sự đo độ liên quan.

**Lấy dư ứng viên trước khi fuse.** Mỗi nhánh lấy `top_k × 4` chứ không phải đúng
`top_k`: RRF chỉ phát huy tác dụng khi nhìn đủ sâu để phát hiện tài liệu được *cả
hai* retriever cùng bình chọn. Nếu mỗi bên chỉ đưa lên `top_k`, phần giao gần như
luôn rỗng và RRF suy biến thành phép nối hai danh sách.

---

## 3. Phân Công Công Việc

Nhóm 5 thành viên — Phương án B (chuyên sâu Retrieval).

| Role | Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|---|---|---|---|---|
| 1 — Team Leader & RAG Architect | | | Điều phối, ghép pipeline chính, Task 9 | ✅ |
| 2 — Data & Dense Search Dev | | | Task 1–3 (thu thập, chuẩn hoá), Task 4 (ChromaDB), Task 5 (semantic search + HyDE) | ✅ |
| 3 — Sparse Search & Advanced Reranking Dev | | | Task 6 (BM25 + TF-IDF), Task 7 (RRF + cross-encoder + MMR), Task 8 (PageIndex fallback) | ✅ |
| 4 — Frontend & Chatbot Dev | | | `app.py` Streamlit, Task 10 (generation có citation) | ✅ |
| 5 — Evaluation & QA Engineer | | | `golden_dataset.json`, `eval_pipeline.py`, `results.md` | ✅ |

---

## 4. Deliverables

### Chatbot
- [x] Giao diện chat Streamlit (`app.py`)
- [x] Trả lời có citation dạng `[Bộ luật Lao động 2019, Điều 25]`
- [x] Hiển thị nguồn: tên file, loại tài liệu, score, bôi vàng từ khoá
- [x] Conversation memory — ghép ngữ cảnh cho câu hỏi follow-up
- [x] Thanh cài đặt: `top_k`, bật/tắt RRF, bật/tắt HyDE, ngưỡng fallback
- [x] Vùng chẩn đoán retrieval (số hit mỗi nhánh, cosine tốt nhất, fallback có chạy không)

### Evaluation
- [x] `evaluation/golden_dataset.json` — **20** cặp Q&A Luật Lao động
- [x] `evaluation/eval_pipeline.py` — RAGAS + tầng metric retrieval tất định
- [x] `evaluation/results.md` — bảng điểm, A/B, worst performers, đề xuất
- [x] So sánh A/B: Hybrid+RRF vs Dense-only

---

## 5. Hướng Dẫn Chạy

```bash
# 1. Cài dependencies
pip install -r requirements.txt

# 2. Tạo .env và điền OPENROUTER_API_KEY
cp .env.example .env

# 3. Index dữ liệu vào ChromaDB (bắt buộc chạy trước, ~5 phút)
python -m src.task4_chunking_indexing

# 4. Chạy chatbot
streamlit run app.py

# 5. Chạy evaluation
#    - A/B tất định trên toàn bộ 20 câu, KHÔNG tốn quota LLM:
python -m group_project.evaluation.eval_pipeline --offline-only
#    - Kèm RAGAS trên 3 câu đầu mỗi config (tốn ~45 lượt gọi LLM):
python -m group_project.evaluation.eval_pipeline --ragas-limit 3

# 6. Chấm điểm pipeline kỹ thuật
pytest tests/test_individual.py -v
```

> **Quota LLM:** OpenRouter free tier giới hạn **50 request/ngày cho cả tài khoản**
> (không phải theo model hay theo API key). RAGAS tiêu ~7–9 lượt gọi cho mỗi câu
> hỏi mỗi config, nên `--ragas-limit` để mặc định là 3. Muốn chạy full 20 câu thì
> cần nạp credit để mở lên 1000 request/ngày.

---

## 6. Lưu ý

Giữ lại repo này nếu học track 3 giai đoạn 2 — dự án sẽ được phát triển tiếp lên
knowledge graph để xử lý các câu hỏi pháp lý nhiều bước.
