# RAG Evaluation Results — Trợ Lý Hỏi Đáp Luật Lao Động

*Sinh tự động bởi `group_project/evaluation/eval_pipeline.py` — 04/08/2026 15:17*

## 1. Thiết lập đánh giá

| Hạng mục | Giá trị |
|---|---|
| Framework | **RAGAS 0.2.15** (4 metric LLM-judged) + tầng metric retrieval tất định |
| Golden dataset | 20 câu hỏi Luật Lao động |
| Corpus | 4 văn bản luật + 10 bài tư vấn → 1069 chunk trong ChromaDB |
| Embedding | `BAAI/bge-m3` (1024 chiều), chunk 800 / overlap 100 |
| LLM sinh câu trả lời | `google/gemma-4-26b-a4b-it:free` qua OpenRouter |
| LLM trọng tài (RAGAS) | cùng model, `temperature=0` |
| top_k khi đánh giá | 3 |

## 2. Hai cấu hình đem so sánh

**A_hybrid_rrf** — Hybrid (Dense + BM25) + RRF  
> semantic_search (bge-m3, cosine) + lexical_search (BM25) → rerank_rrf(k=60) → fallback PageIndex khi cosine gốc < 0.48

**B_dense_only** — Dense-only (baseline)  
> Chỉ semantic_search trên ChromaDB, không BM25, không RRF

## 3. Kết quả chính — Metric retrieval (toàn bộ dataset)

Chấm bằng đối chiếu số hiệu điều luật giữa chunk truy hồi được và `expected_context` của golden dataset. Không dùng LLM nên chạy được trên **toàn bộ** câu hỏi, và kết quả tất định — chạy lại cho ra đúng con số cũ.

| Metric | Hybrid (Dense + BM25) + RRF | Dense-only (baseline) | Δ (A−B) |
|---|---|---|---|
| Hit Rate@k — có lấy được điều luật đúng không | **0.6500** | 0.6000 | +0.0500 |
| Article Recall — lấy đủ bao nhiêu % điều luật cần | **0.6000** | 0.5500 | +0.0500 |
| Article Precision — % chunk thực sự liên quan | **0.2667** | 0.2333 | +0.0334 |
| MRR — điều luật đúng nằm ở hạng bao nhiêu | **0.5000** | 0.4750 | +0.0250 |
| **Trung bình** | **0.5042** | 0.4646 | +0.0396 |

## 4. Kết quả RAGAS (LLM-judged)

> **Chưa chạy được ở lần này — hết quota LLM.** OpenRouter free tier giới hạn **50 request/ngày cho cả tài khoản** (không phải theo model hay theo API key: đổi sang model `:free` khác hoặc tạo key mới đều không reset). Quota đã bị tiêu hết trong quá trình phát triển và kiểm thử pipeline, nên lượt chấm RAGAS bị chặn bởi lỗi:

> ```
> 429 Rate limit exceeded: free-models-per-day
> X-RateLimit-Limit: 50   X-RateLimit-Remaining: 0
> ```

> Phần đường dây RAGAS **đã được kiểm chứng chạy đúng** trước khi hết quota (một mẫu thử `Faithfulness` trả về `0.5000`), nên đây là giới hạn hạ tầng chứ không phải lỗi code. Chạy lại sau khi quota reset:

> ```bash
> python -m group_project.evaluation.eval_pipeline --ragas-limit 3
> ```

> Vì vậy bảng ở mục 3 là căn cứ đánh giá chính của báo cáo này. Điều đó cũng cho thấy giá trị của việc thiết kế sẵn tầng metric tất định: khi phần LLM-judged không chạy được vì lý do ngoài tầm kiểm soát, việc so sánh A/B vẫn tiến hành được trên toàn bộ 20 câu hỏi.

## 5. Worst performers — 3 câu tệ nhất (config A)

| # | Câu hỏi | Điều luật cần | Điều luật lấy được | Recall | MRR |
|---|---|---|---|---|---|
| 1 | Công ty sa thải tôi qua tin nhắn Zalo mà không báo trước 30 ngày thì c… | Đ122, Đ125 | *(không có)* | 0.0 | 0.0 |
| 2 | Một tháng tôi bị bắt OT tối đa bao nhiêu giờ, và một năm tối đa bao nh… | Đ107 | Đ105, Đ106 | 0.0 | 0.0 |
| 3 | Tôi mới làm được 7 tháng rồi nghỉ việc, phép năm chưa nghỉ hết có được… | Đ113 | *(không có)* | 0.0 | 0.0 |

### Phân tích nguyên nhân

- **7/20 câu trượt hoàn toàn** (không chunk nào *mang tiêu đề* điều luật cần). Nguyên nhân là **chunking cắt ngang điều luật**: với `chunk_size=800`, một điều dài bị xé làm nhiều mảnh và chỉ mảnh ĐẦU TIÊN giữ được dòng tiêu đề `Điều N.`.

  Đã kiểm chứng trực tiếp trên Q06 (*'một tháng OT tối đa bao nhiêu giờ'*): nội dung trả lời nằm ở chunk #137, chứa đúng câu *'không quá 200 giờ trong 01 năm'* — nhưng chunk đó **không** có dòng `Điều 107.` (dòng này rơi vào chunk #136, mà chunk #136 lại bị cắt trước khi tới các con số). Retriever thực tế **đã lấy về nội dung đúng**, chỉ là không có nhãn điều luật để metric ghi nhận.

  > ⚠ **Giới hạn của phép đo — cần nói rõ:** metric này đo *'có lấy được chunk được gắn nhãn đúng điều luật không'*, không phải *'có lấy được nội dung trả lời được câu hỏi không'*. Vì vậy con số tuyệt đối bị **đánh giá thấp hơn** năng lực thật của retriever. Điều quan trọng: sai lệch này tác động **như nhau lên cả hai config**, nên phép so sánh A/B ở mục 3 vẫn có giá trị — chỉ không nên đọc `hit_rate = 0.65` như 'hệ thống trả lời sai 35% số câu'. Muốn đo đúng chất lượng trả lời thì phải dùng faithfulness / answer_relevancy của RAGAS (mục 4).
- **2 câu chỉ lấy được một phần** số điều luật cần. Đây là các câu hỏi bắc cầu hai văn bản — ví dụ hỏi vừa thời gian thử việc (Điều 25) vừa lương thử việc (Điều 26): retriever kéo về đúng một trong hai rồi các slot top_k còn lại bị chiếm bởi chunk lân cận của cùng điều đó.
- **Article Precision thấp (~0.27) là do trần lý thuyết, không phải lỗi retriever**: mỗi câu hỏi chỉ cần 1–2 điều luật trong khi top_k=3, nên precision tối đa đạt được đã là ~0.33–0.67. Con số này chỉ có ý nghĩa khi so A với B, không nên đọc như tỷ lệ tuyệt đối.

## 6. Kết luận A/B

**Hybrid (Dense + BM25) + RRF thắng 4/4 metric retrieval.** Trung bình 0.5042 so với 0.4646 (+0.0396).

Mức chênh khiêm tốn nhưng nhất quán, và cơ chế đứng sau nó giải thích được: nhánh BM25 cứu đúng loại truy vấn mà dense embedding hay trượt — câu hỏi neo vào **số hiệu điều luật và con số** ('Điều 25', '60 ngày', '85%', 'vùng I'). Embedding coi các con số này gần như nhiễu, còn BM25 thì khớp chính xác. Ngược lại, câu hỏi diễn đạt đời thường ('bị đuổi việc qua Zalo') thì dense thắng vì không có từ khoá nào trùng mặt chữ với văn bản luật.

Đó chính là lý do RRF đáng giá ở bài toán này: hai retriever mạnh ở hai lớp truy vấn khác nhau, và tài liệu được **cả hai** cùng bình chọn sẽ được RRF đẩy lên đầu.

## 7. Đề xuất cải tiến

### Cải tiến 1 — Chunking theo cấu trúc điều luật thay vì cắt theo ký tự
**Vấn đề:** `RecursiveCharacterTextSplitter(800/100)` cắt ngang giữa điều luật, làm mất liên kết giữa tiêu đề `Điều N.` và các khoản bên dưới.  
**Action:** tách bằng regex `^Điều\s+\d+\.` để mỗi điều là một chunk, điều dài thì cắt tiếp theo khoản nhưng **lặp lại dòng tiêu đề** ở đầu mỗi mảnh con.  
**Kỳ vọng:** đánh trực tiếp vào nhóm câu trượt hoàn toàn ở mục 5 — đây là thay đổi có đòn bẩy lớn nhất trong danh sách này.

### Cải tiến 2 — Đưa số hiệu điều luật vào metadata và cho phép lọc
**Vấn đề:** metadata hiện chỉ có `source`, `type`, `chunk_index`; câu hỏi trích dẫn thẳng 'Điều 25' vẫn phải đi đường vòng qua tìm kiếm toàn văn.  
**Action:** khi index, rút `Điều N` + tên văn bản vào metadata; khi truy vấn có chứa số hiệu điều luật thì thêm bước lọc metadata trước khi xếp hạng.  
**Kỳ vọng:** cải thiện MRR cho nhóm câu hỏi tra cứu trực tiếp, vốn là kiểu hỏi rất phổ biến khi người dùng đã cầm sẵn hợp đồng trên tay.

### Cải tiến 3 — Bổ sung cross-encoder rerank sau RRF
**Vấn đề:** RRF chỉ gộp *thứ hạng*, không đọc nội dung, nên không phân biệt được hai chunk cùng hạng nhưng khác hẳn nhau về mức độ trả lời trúng câu hỏi.  
**Action:** `rerank_cross_encoder()` (Task 7) đã viết sẵn — bật lên chấm lại top 10 sau RRF rồi mới cắt xuống top_k.  
**Kỳ vọng:** tăng Article Precision và faithfulness, đổi lại thêm độ trễ; nên đo lại A/B trước khi bật mặc định.

## 8. Phụ lục — bảng điểm đầy đủ

<details>
<summary>Điểm từng câu (config A)</summary>

| ID | Chủ đề | Hit | Recall | Precision | MRR |
|---|---|---|---|---|---|
| Q01 | thử việc | 1 | 1.0 | 0.6667 | 1.0 |
| Q02 | sa thải | 0 | 0.0 | 0.0 | 0.0 |
| Q03 | thử việc | 1 | 1.0 | 0.3333 | 1.0 |
| Q04 | thử việc | 1 | 0.5 | 0.3333 | 1.0 |
| Q05 | thử việc | 1 | 1.0 | 0.3333 | 0.3333 |
| Q06 | làm thêm giờ | 0 | 0.0 | 0.0 | 0.0 |
| Q07 | làm thêm giờ | 1 | 0.5 | 0.3333 | 1.0 |
| Q08 | lương làm thêm | 1 | 1.0 | 0.3333 | 0.3333 |
| Q09 | lương làm thêm | 1 | 1.0 | 0.6667 | 1.0 |
| Q10 | nghỉ phép | 1 | 1.0 | 0.3333 | 0.5 |
| Q11 | nghỉ phép | 0 | 0.0 | 0.0 | 0.0 |
| Q12 | nghỉ phép | 0 | 0.0 | 0.0 | 0.0 |
| Q13 | nghỉ việc riêng | 1 | 1.0 | 0.3333 | 1.0 |
| Q14 | sa thải | 0 | 0.0 | 0.0 | 0.0 |
| Q15 | sa thải | 0 | 0.0 | 0.0 | 0.0 |
| Q16 | chấm dứt hợp đồng | 1 | 1.0 | 0.3333 | 0.5 |
| Q17 | chấm dứt hợp đồng | 0 | 0.0 | 0.0 | 0.0 |
| Q18 | chấm dứt hợp đồng trái luật | 1 | 1.0 | 0.3333 | 1.0 |
| Q19 | lương tối thiểu | 1 | 1.0 | 0.6667 | 1.0 |
| Q20 | loại hợp đồng | 1 | 1.0 | 0.3333 | 0.3333 |

</details>
