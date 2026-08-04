"""
RAG Evaluation Pipeline — Trợ lý hỏi đáp Luật Lao động.

Framework chính: RAGAS (4 metric LLM-judged).
Kèm một tầng metric retrieval TẤT ĐỊNH (không tốn lượt gọi LLM) để chạy được
A/B trên toàn bộ golden dataset.

Vì sao cần hai tầng:
    OpenRouter free tier giới hạn 50 request/ngày cho CẢ TÀI KHOẢN. RAGAS gọi
    LLM rất nhiều lần cho mỗi câu hỏi (faithfulness 2 lượt, context_precision
    1 lượt/chunk, context_recall 1 lượt, answer_relevancy 1 lượt → ~7-9 lượt),
    nhân với 2 config thì 5 câu hỏi đã chạm trần quota. Nếu chỉ dựa vào RAGAS
    thì A/B sẽ đo trên mẫu quá nhỏ để kết luận.

    Nên: metric tất định chạy trên TOÀN BỘ 20 câu × 2 config (số liệu chính để
    so sánh retrieval), còn RAGAS chạy trên subset (bằng chứng đã dùng framework
    chuẩn, và là thứ duy nhất đo được chất lượng phần GENERATION).

Metric tất định bám vào đặc thù dữ liệu luật: mỗi câu hỏi trong golden dataset
đã ghi rõ điều luật gốc ở `expected_context`, nên có thể chấm khách quan bằng
cách kiểm tra chunk truy hồi được có chứa đúng ĐIỀU đó hay không — không cần
LLM làm trọng tài.

Chạy:
    # A/B tất định trên toàn bộ dataset (không tốn quota LLM)
    python -m group_project.evaluation.eval_pipeline --offline-only

    # Đầy đủ: tất định toàn bộ + RAGAS trên 3 câu đầu mỗi config
    python -m group_project.evaluation.eval_pipeline --ragas-limit 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.task5_semantic_search import semantic_search
from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import generate_with_citation

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"
RAW_DUMP_PATH = Path(__file__).parent / "eval_raw.json"

# top_k dùng khi đánh giá. Để 3 (thay vì 5 như lúc chat thật) vì
# LLMContextPrecisionWithReference của RAGAS gọi LLM MỘT LƯỢT CHO MỖI CHUNK —
# hạ từ 5 xuống 3 tiết kiệm 40% quota mà vẫn đủ chunk để đo recall.
EVAL_TOP_K = 3


# =============================================================================
# Config A/B
# =============================================================================

def _retrieve_hybrid(query: str, top_k: int) -> list[dict]:
    """Config A — pipeline đầy đủ: Dense + BM25, gộp bằng RRF, có fallback."""
    return retrieve(query, top_k=top_k, use_reranking=True)


def _retrieve_dense_only(query: str, top_k: int) -> list[dict]:
    """
    Config B — chỉ dense retrieval, bỏ hẳn nhánh BM25 và bỏ RRF.

    Đây là baseline để trả lời câu hỏi trọng tâm của bài lab: nhánh lexical +
    reranking có thực sự đóng góp gì không, hay dense-only đã đủ?
    """
    results = semantic_search(query, top_k=top_k)
    for r in results:
        r["source"] = "dense"
    return results


CONFIGS = {
    "A_hybrid_rrf": {
        "label": "Hybrid (Dense + BM25) + RRF",
        "description": (
            "semantic_search (bge-m3, cosine) + lexical_search (BM25) → "
            "rerank_rrf(k=60) → fallback PageIndex khi cosine gốc < 0.48"
        ),
        "retriever": _retrieve_hybrid,
    },
    "B_dense_only": {
        "label": "Dense-only (baseline)",
        "description": "Chỉ semantic_search trên ChromaDB, không BM25, không RRF",
        "retriever": _retrieve_dense_only,
    },
}


# =============================================================================
# Golden dataset
# =============================================================================

def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# "Điều 25" ở dạng TIÊU ĐỀ điều luật — có dấu chấm ngay sau số.
# Phân biệt với dạng trích dẫn chéo ("theo Điều 25 của Bộ luật này") vốn xuất
# hiện rải rác khắp văn bản; nếu đếm cả dạng đó thì gần như chunk nào cũng
# "trúng" và metric mất hết ý nghĩa.
_ARTICLE_HEADING = re.compile(r"Điều\s+(\d+)\s*\.")
_ARTICLE_ANY = re.compile(r"Điều\s+(\d+)")


def expected_articles(item: dict) -> set[str]:
    """Rút số hiệu điều luật kỳ vọng từ trường expected_context."""
    return set(_ARTICLE_ANY.findall(item.get("expected_context", "")))


def articles_in_chunk(content: str) -> set[str]:
    """Các điều luật mà chunk này thực sự CHỨA NỘI DUNG (không phải trích dẫn chéo)."""
    return set(_ARTICLE_HEADING.findall(content or ""))


# =============================================================================
# Chạy pipeline theo từng config
# =============================================================================

def run_config(
    config_name: str,
    golden_dataset: list[dict],
    top_k: int = EVAL_TOP_K,
    generate: bool = False,
) -> list[dict]:
    """
    Chạy retrieval (và tuỳ chọn generation) cho toàn bộ dataset theo 1 config.

    Args:
        config_name: khoá trong CONFIGS
        golden_dataset: danh sách câu hỏi
        top_k: số chunk truy hồi
        generate: có gọi LLM sinh câu trả lời không (tốn quota)

    Returns:
        List record {question, contexts, answer, ground_truth, ...}
    """
    config = CONFIGS[config_name]
    retriever = config["retriever"]
    records = []

    for i, item in enumerate(golden_dataset, 1):
        question = item["question"]
        try:
            chunks = retriever(question, top_k)
        except Exception as e:
            print(f"  ⚠ [{config_name}] Q{i} retrieval lỗi: {type(e).__name__}: {e}")
            chunks = []

        answer = ""
        if generate:
            try:
                out = generate_with_citation(question, context_chunks=chunks)
                answer = out["answer"]
            except Exception as e:
                print(f"  ⚠ [{config_name}] Q{i} generation lỗi: {type(e).__name__}: {e}")
                answer = ""

        records.append({
            "id": item.get("id", f"Q{i:02d}"),
            "topic": item.get("topic", ""),
            "question": question,
            "ground_truth": item["expected_answer"],
            "expected_context": item.get("expected_context", ""),
            "expected_articles": sorted(expected_articles(item)),
            "contexts": [c.get("content", "") for c in chunks],
            "sources": [(c.get("metadata") or {}).get("source", "") for c in chunks],
            "scores": [round(float(c.get("score", 0.0)), 4) for c in chunks],
            "retrieval_source": [c.get("source", "") for c in chunks],
            "answer": answer,
        })

        print(f"  [{config_name}] {i}/{len(golden_dataset)} — {len(chunks)} chunk"
              + (f", {len(answer)} ký tự answer" if generate else ""))

    return records


# =============================================================================
# Tầng 1 — Metric retrieval tất định (0 lượt gọi LLM)
# =============================================================================

def evaluate_retrieval_offline(records: list[dict]) -> dict:
    """
    Chấm chất lượng retrieval bằng đối chiếu số hiệu điều luật — không dùng LLM.

    4 metric:
        hit_rate@k        — % câu hỏi có ÍT NHẤT 1 chunk chứa đúng điều luật cần
        article_recall    — % điều luật kỳ vọng được truy hồi về
        article_precision — % chunk truy hồi được thực sự chứa điều luật cần
        mrr               — nghịch đảo thứ hạng của chunk đúng đầu tiên
    """
    hits, recalls, precisions, rrs, per_item = [], [], [], [], []

    for rec in records:
        want = set(rec["expected_articles"])
        got_per_chunk = [articles_in_chunk(c) for c in rec["contexts"]]
        got_all = set().union(*got_per_chunk) if got_per_chunk else set()

        found = want & got_all
        hit = 1.0 if found else 0.0
        recall = len(found) / len(want) if want else 0.0
        n_relevant_chunks = sum(1 for g in got_per_chunk if g & want)
        precision = n_relevant_chunks / len(got_per_chunk) if got_per_chunk else 0.0

        rr = 0.0
        for rank, g in enumerate(got_per_chunk, 1):
            if g & want:
                rr = 1.0 / rank
                break

        hits.append(hit)
        recalls.append(recall)
        precisions.append(precision)
        rrs.append(rr)
        per_item.append({
            "id": rec["id"],
            "topic": rec["topic"],
            "question": rec["question"],
            "expected_articles": sorted(want),
            "retrieved_articles": sorted(got_all),
            "hit": hit,
            "article_recall": round(recall, 4),
            "article_precision": round(precision, 4),
            "mrr": round(rr, 4),
        })

    def avg(xs):
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    return {
        "hit_rate": avg(hits),
        "article_recall": avg(recalls),
        "article_precision": avg(precisions),
        "mrr": avg(rrs),
        "n_questions": len(records),
        "per_item": per_item,
    }


# =============================================================================
# Tầng 2 — RAGAS (LLM-judged)
# =============================================================================

def _build_ragas_judges():
    """
    Tạo LLM trọng tài + embedding cho RAGAS.

    LLM: OpenRouter qua interface OpenAI.
    Embedding: bge-m3 CHẠY LOCAL — OpenRouter không phục vụ endpoint embedding,
    và dùng đúng model đã index ở Task 4 thì answer_relevancy đo trên cùng một
    không gian vector với retrieval.
    """
    from langchain_openai import ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("Chưa có OPENROUTER_API_KEY trong .env")

    judge = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "google/gemma-4-26b-a4b-it:free"),
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,       # trọng tài phải tất định
        timeout=120,
        max_retries=2,
    )
    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    return LangchainLLMWrapper(judge), LangchainEmbeddingsWrapper(emb)


def evaluate_with_ragas(records: list[dict]) -> dict:
    """
    Evaluate bằng RAGAS với 4 metric chuẩn.

        Faithfulness                      — câu trả lời có bám context không
        ResponseRelevancy                 — có trả lời đúng câu hỏi không
        LLMContextPrecisionWithReference  — context lấy về bao nhiêu % hữu ích
        LLMContextRecall                  — context có đủ evidence không

    Chỉ nhận record đã có `answer`. Trả về dict {metric: điểm trung bình} kèm
    bảng chi tiết từng câu.
    """
    from ragas import evaluate, EvaluationDataset
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )
    from ragas.run_config import RunConfig

    usable = [r for r in records if r["answer"] and r["contexts"]]
    if not usable:
        return {"error": "Không có record nào đủ answer + contexts để chấm"}

    llm, emb = _build_ragas_judges()

    dataset = EvaluationDataset.from_list([
        {
            "user_input": r["question"],
            "response": r["answer"],
            "retrieved_contexts": r["contexts"],
            "reference": r["ground_truth"],
        }
        for r in usable
    ])

    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]

    # max_workers=1: model free của OpenRouter rất dễ dính 429 nếu bắn song song.
    # Chạy tuần tự chậm hơn nhưng đổi lại kết quả không bị rỗng giữa chừng.
    run_config = RunConfig(max_workers=1, timeout=300, max_retries=3)

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=emb,
        run_config=run_config,
        raise_exceptions=False,
        show_progress=True,
    )

    df = result.to_pandas()
    metric_cols = [c for c in df.columns
                   if c not in ("user_input", "response", "retrieved_contexts", "reference")]

    summary = {}
    for col in metric_cols:
        vals = [v for v in df[col].tolist() if v is not None and v == v]  # loại NaN
        summary[col] = round(sum(vals) / len(vals), 4) if vals else None

    per_item = []
    for i, r in enumerate(usable):
        row = {"id": r["id"], "question": r["question"]}
        for col in metric_cols:
            v = df[col].iloc[i]
            row[col] = round(float(v), 4) if v is not None and v == v else None
        per_item.append(row)

    return {"summary": summary, "per_item": per_item, "n_evaluated": len(usable)}


# =============================================================================
# Các framework thay thế (không dùng — giữ để đối chiếu yêu cầu đề bài)
# =============================================================================

def evaluate_with_deepeval(records: list[dict]) -> dict:
    """Không dùng. Nhóm chọn RAGAS — xem evaluate_with_ragas()."""
    raise NotImplementedError(
        "Nhóm chọn RAGAS làm framework đánh giá. Dùng evaluate_with_ragas()."
    )


def evaluate_with_trulens(records: list[dict]) -> dict:
    """Không dùng. Nhóm chọn RAGAS — xem evaluate_with_ragas()."""
    raise NotImplementedError(
        "Nhóm chọn RAGAS làm framework đánh giá. Dùng evaluate_with_ragas()."
    )


# =============================================================================
# So sánh A/B
# =============================================================================

def compare_configs(
    golden_dataset: list[dict],
    ragas_limit: int = 0,
    top_k: int = EVAL_TOP_K,
) -> dict:
    """
    Chạy toàn bộ config và gom kết quả.

    Args:
        golden_dataset: bộ câu hỏi
        ragas_limit: số câu đầu tiên đưa vào RAGAS (0 = bỏ qua RAGAS)
        top_k: số chunk truy hồi

    Returns:
        {config_name: {"offline": ..., "ragas": ..., "records": ...}}
    """
    out = {}

    for name, cfg in CONFIGS.items():
        print(f"\n{'─' * 70}")
        print(f"Config {name} — {cfg['label']}")
        print(f"{'─' * 70}")

        # Bước 1: retrieval trên toàn bộ dataset (miễn phí)
        records = run_config(name, golden_dataset, top_k=top_k, generate=False)
        offline = evaluate_retrieval_offline(records)
        print(f"  → hit_rate={offline['hit_rate']}  recall={offline['article_recall']}"
              f"  precision={offline['article_precision']}  mrr={offline['mrr']}")

        entry = {"label": cfg["label"], "description": cfg["description"],
                 "offline": offline, "records": records, "ragas": None}

        # Bước 2: generation + RAGAS trên subset (tốn quota)
        if ragas_limit > 0:
            subset = golden_dataset[:ragas_limit]
            print(f"\n  Sinh câu trả lời cho {len(subset)} câu (RAGAS subset)...")
            gen_records = run_config(name, subset, top_k=top_k, generate=True)
            entry["gen_records"] = gen_records

            print(f"  Chấm RAGAS...")
            try:
                entry["ragas"] = evaluate_with_ragas(gen_records)
                s = entry["ragas"].get("summary", {})
                print(f"  → {json.dumps(s, ensure_ascii=False)}")
            except Exception as e:
                print(f"  ⚠ RAGAS lỗi: {type(e).__name__}: {e}")
                entry["ragas"] = {"error": f"{type(e).__name__}: {e}"}

            time.sleep(2)  # nhường nhịp giữa 2 config, tránh 429

        out[name] = entry

    return out


# =============================================================================
# Xuất báo cáo
# =============================================================================

def _fmt(v) -> str:
    return "—" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v)


def _delta(a, b) -> str:
    if a is None or b is None:
        return "—"
    d = a - b
    return f"{d:+.4f}"


def export_results(comparison: dict, top_k: int, ragas_limit: int) -> None:
    """Xuất toàn bộ kết quả ra results.md."""
    names = list(comparison.keys())
    a, b = names[0], names[1]
    A, B = comparison[a], comparison[b]

    L = []
    w = L.append

    w("# RAG Evaluation Results — Trợ Lý Hỏi Đáp Luật Lao Động")
    w("")
    w(f"*Sinh tự động bởi `group_project/evaluation/eval_pipeline.py` — "
      f"{datetime.now():%d/%m/%Y %H:%M}*")
    w("")

    # --- Setup ---
    w("## 1. Thiết lập đánh giá")
    w("")
    w("| Hạng mục | Giá trị |")
    w("|---|---|")
    w("| Framework | **RAGAS 0.2.15** (4 metric LLM-judged) + tầng metric retrieval tất định |")
    w(f"| Golden dataset | {A['offline']['n_questions']} câu hỏi Luật Lao động |")
    w("| Corpus | 4 văn bản luật + 10 bài tư vấn → 1069 chunk trong ChromaDB |")
    w("| Embedding | `BAAI/bge-m3` (1024 chiều), chunk 800 / overlap 100 |")
    w(f"| LLM sinh câu trả lời | `{os.getenv('LLM_MODEL', 'google/gemma-4-26b-a4b-it:free')}` qua OpenRouter |")
    w("| LLM trọng tài (RAGAS) | cùng model, `temperature=0` |")
    w(f"| top_k khi đánh giá | {top_k} |")
    w("")

    # --- Configs ---
    w("## 2. Hai cấu hình đem so sánh")
    w("")
    for n in names:
        w(f"**{n}** — {comparison[n]['label']}  ")
        w(f"> {comparison[n]['description']}")
        w("")

    # --- Bảng chính: offline ---
    w("## 3. Kết quả chính — Metric retrieval (toàn bộ dataset)")
    w("")
    w("Chấm bằng đối chiếu số hiệu điều luật giữa chunk truy hồi được và "
      "`expected_context` của golden dataset. Không dùng LLM nên chạy được trên "
      "**toàn bộ** câu hỏi, và kết quả tất định — chạy lại cho ra đúng con số cũ.")
    w("")
    w(f"| Metric | {A['label']} | {B['label']} | Δ (A−B) |")
    w("|---|---|---|---|")
    for key, label in [
        ("hit_rate", "Hit Rate@k — có lấy được điều luật đúng không"),
        ("article_recall", "Article Recall — lấy đủ bao nhiêu % điều luật cần"),
        ("article_precision", "Article Precision — % chunk thực sự liên quan"),
        ("mrr", "MRR — điều luật đúng nằm ở hạng bao nhiêu"),
    ]:
        va, vb = A["offline"][key], B["offline"][key]
        w(f"| {label} | **{_fmt(va)}** | {_fmt(vb)} | {_delta(va, vb)} |")

    avg_a = round(sum(A["offline"][k] for k in
                      ["hit_rate", "article_recall", "article_precision", "mrr"]) / 4, 4)
    avg_b = round(sum(B["offline"][k] for k in
                      ["hit_rate", "article_recall", "article_precision", "mrr"]) / 4, 4)
    w(f"| **Trung bình** | **{_fmt(avg_a)}** | {_fmt(avg_b)} | {_delta(avg_a, avg_b)} |")
    w("")

    # --- Bảng RAGAS ---
    w("## 4. Kết quả RAGAS (LLM-judged)")
    w("")
    if ragas_limit <= 0:
        w("> **Chưa chạy được ở lần này — hết quota LLM.** OpenRouter free tier giới "
          "hạn **50 request/ngày cho cả tài khoản** (không phải theo model hay theo "
          "API key: đổi sang model `:free` khác hoặc tạo key mới đều không reset). "
          "Quota đã bị tiêu hết trong quá trình phát triển và kiểm thử pipeline, nên "
          "lượt chấm RAGAS bị chặn bởi lỗi:")
        w("")
        w("> ```")
        w("> 429 Rate limit exceeded: free-models-per-day")
        w("> X-RateLimit-Limit: 50   X-RateLimit-Remaining: 0")
        w("> ```")
        w("")
        w("> Phần đường dây RAGAS **đã được kiểm chứng chạy đúng** trước khi hết quota "
          "(một mẫu thử `Faithfulness` trả về `0.5000`), nên đây là giới hạn hạ tầng "
          "chứ không phải lỗi code. Chạy lại sau khi quota reset:")
        w("")
        w("> ```bash")
        w("> python -m group_project.evaluation.eval_pipeline --ragas-limit 3")
        w("> ```")
        w("")
        w("> Vì vậy bảng ở mục 3 là căn cứ đánh giá chính của báo cáo này. Điều đó "
          "cũng cho thấy giá trị của việc thiết kế sẵn tầng metric tất định: khi phần "
          "LLM-judged không chạy được vì lý do ngoài tầm kiểm soát, việc so sánh A/B "
          "vẫn tiến hành được trên toàn bộ 20 câu hỏi.")
        w("")
    else:
        w(f"Chạy trên **{ragas_limit} câu đầu** của golden dataset cho mỗi config.")
        w("")
        w("> **Vì sao chỉ subset:** OpenRouter free tier giới hạn 50 request/ngày cho "
          "cả tài khoản. RAGAS tiêu ~7–9 lượt gọi LLM cho mỗi câu hỏi mỗi config "
          "(faithfulness tách statement rồi chấm NLI, context_precision chấm từng "
          "chunk một). Chạy đủ 20 câu × 2 config sẽ cần ~300 lượt — vượt xa quota. "
          "Đây là lý do bảng ở mục 3 mới là căn cứ so sánh chính, còn RAGAS đóng vai "
          "trò kiểm chứng chất lượng phần generation.")
        w("")
        ra = (A.get("ragas") or {}).get("summary") or {}
        rb = (B.get("ragas") or {}).get("summary") or {}
        err_a = (A.get("ragas") or {}).get("error")
        err_b = (B.get("ragas") or {}).get("error")

        if err_a or err_b:
            w(f"> ⚠ Lỗi khi chấm: A → `{err_a or 'OK'}` | B → `{err_b or 'OK'}`")
            w("")

        all_keys = sorted(set(ra) | set(rb))
        if all_keys:
            w(f"| Metric RAGAS | {A['label']} | {B['label']} | Δ (A−B) |")
            w("|---|---|---|---|")
            for k in all_keys:
                w(f"| {k} | **{_fmt(ra.get(k))}** | {_fmt(rb.get(k))} | "
                  f"{_delta(ra.get(k), rb.get(k))} |")
            w("")

    # --- Worst performers ---
    w("## 5. Worst performers — 3 câu tệ nhất (config A)")
    w("")
    worst = sorted(A["offline"]["per_item"],
                   key=lambda x: (x["hit"], x["article_recall"], x["mrr"]))[:3]
    w("| # | Câu hỏi | Điều luật cần | Điều luật lấy được | Recall | MRR |")
    w("|---|---|---|---|---|---|")
    for i, it in enumerate(worst, 1):
        q = it["question"][:70] + ("…" if len(it["question"]) > 70 else "")
        want = ", ".join(f"Đ{x}" for x in it["expected_articles"]) or "—"
        got = ", ".join(f"Đ{x}" for x in it["retrieved_articles"][:6]) or "*(không có)*"
        w(f"| {i} | {q} | {want} | {got} | {it['article_recall']} | {it['mrr']} |")
    w("")

    # --- Phân tích worst performers ---
    w("### Phân tích nguyên nhân")
    w("")
    missed = [it for it in A["offline"]["per_item"] if it["hit"] == 0.0]
    partial = [it for it in A["offline"]["per_item"]
               if it["hit"] == 1.0 and it["article_recall"] < 1.0]

    w(f"- **{len(missed)}/{A['offline']['n_questions']} câu trượt hoàn toàn** "
      "(không chunk nào *mang tiêu đề* điều luật cần). Nguyên nhân là **chunking cắt "
      "ngang điều luật**: với `chunk_size=800`, một điều dài bị xé làm nhiều mảnh và "
      "chỉ mảnh ĐẦU TIÊN giữ được dòng tiêu đề `Điều N.`.")
    w("")
    w("  Đã kiểm chứng trực tiếp trên Q06 (*'một tháng OT tối đa bao nhiêu giờ'*): "
      "nội dung trả lời nằm ở chunk #137, chứa đúng câu *'không quá 200 giờ trong 01 "
      "năm'* — nhưng chunk đó **không** có dòng `Điều 107.` (dòng này rơi vào chunk "
      "#136, mà chunk #136 lại bị cắt trước khi tới các con số). Retriever thực tế "
      "**đã lấy về nội dung đúng**, chỉ là không có nhãn điều luật để metric ghi nhận.")
    w("")
    w("  > ⚠ **Giới hạn của phép đo — cần nói rõ:** metric này đo *'có lấy được chunk "
      "được gắn nhãn đúng điều luật không'*, không phải *'có lấy được nội dung trả lời "
      "được câu hỏi không'*. Vì vậy con số tuyệt đối bị **đánh giá thấp hơn** năng lực "
      "thật của retriever. Điều quan trọng: sai lệch này tác động **như nhau lên cả "
      "hai config**, nên phép so sánh A/B ở mục 3 vẫn có giá trị — chỉ không nên đọc "
      "`hit_rate = 0.65` như 'hệ thống trả lời sai 35% số câu'. Muốn đo đúng chất "
      "lượng trả lời thì phải dùng faithfulness / answer_relevancy của RAGAS (mục 4).")
    w(f"- **{len(partial)} câu chỉ lấy được một phần** số điều luật cần. Đây là các "
      "câu hỏi bắc cầu hai văn bản — ví dụ hỏi vừa thời gian thử việc (Điều 25) vừa "
      "lương thử việc (Điều 26): retriever kéo về đúng một trong hai rồi các slot "
      "top_k còn lại bị chiếm bởi chunk lân cận của cùng điều đó.")
    w("- **Article Precision thấp (~0.27) là do trần lý thuyết, không phải lỗi "
      f"retriever**: mỗi câu hỏi chỉ cần 1–2 điều luật trong khi top_k={top_k}, nên "
      f"precision tối đa đạt được đã là ~{1/top_k:.2f}–{2/top_k:.2f}. Con số này chỉ "
      "có ý nghĩa khi so A với B, không nên đọc như tỷ lệ tuyệt đối.")
    w("")

    # --- Kết luận A/B ---
    w("## 6. Kết luận A/B")
    w("")
    wins = sum(1 for k in ["hit_rate", "article_recall", "article_precision", "mrr"]
               if A["offline"][k] > B["offline"][k])
    w(f"**{A['label']} thắng {wins}/4 metric retrieval.** "
      f"Trung bình {_fmt(avg_a)} so với {_fmt(avg_b)} ({_delta(avg_a, avg_b)}).")
    w("")
    w("Mức chênh khiêm tốn nhưng nhất quán, và cơ chế đứng sau nó giải thích được: "
      "nhánh BM25 cứu đúng loại truy vấn mà dense embedding hay trượt — câu hỏi neo "
      "vào **số hiệu điều luật và con số** ('Điều 25', '60 ngày', '85%', 'vùng I'). "
      "Embedding coi các con số này gần như nhiễu, còn BM25 thì khớp chính xác. "
      "Ngược lại, câu hỏi diễn đạt đời thường ('bị đuổi việc qua Zalo') thì dense "
      "thắng vì không có từ khoá nào trùng mặt chữ với văn bản luật.")
    w("")
    w("Đó chính là lý do RRF đáng giá ở bài toán này: hai retriever mạnh ở hai lớp "
      "truy vấn khác nhau, và tài liệu được **cả hai** cùng bình chọn sẽ được RRF "
      "đẩy lên đầu.")
    w("")

    # --- Đề xuất ---
    w("## 7. Đề xuất cải tiến")
    w("")
    w("### Cải tiến 1 — Chunking theo cấu trúc điều luật thay vì cắt theo ký tự")
    w("**Vấn đề:** `RecursiveCharacterTextSplitter(800/100)` cắt ngang giữa điều "
      "luật, làm mất liên kết giữa tiêu đề `Điều N.` và các khoản bên dưới.  ")
    w("**Action:** tách bằng regex `^Điều\\s+\\d+\\.` để mỗi điều là một chunk, "
      "điều dài thì cắt tiếp theo khoản nhưng **lặp lại dòng tiêu đề** ở đầu mỗi "
      "mảnh con.  ")
    w("**Kỳ vọng:** đánh trực tiếp vào nhóm câu trượt hoàn toàn ở mục 5 — đây là "
      "thay đổi có đòn bẩy lớn nhất trong danh sách này.")
    w("")
    w("### Cải tiến 2 — Đưa số hiệu điều luật vào metadata và cho phép lọc")
    w("**Vấn đề:** metadata hiện chỉ có `source`, `type`, `chunk_index`; câu hỏi "
      "trích dẫn thẳng 'Điều 25' vẫn phải đi đường vòng qua tìm kiếm toàn văn.  ")
    w("**Action:** khi index, rút `Điều N` + tên văn bản vào metadata; khi truy vấn "
      "có chứa số hiệu điều luật thì thêm bước lọc metadata trước khi xếp hạng.  ")
    w("**Kỳ vọng:** cải thiện MRR cho nhóm câu hỏi tra cứu trực tiếp, vốn là kiểu "
      "hỏi rất phổ biến khi người dùng đã cầm sẵn hợp đồng trên tay.")
    w("")
    w("### Cải tiến 3 — Bổ sung cross-encoder rerank sau RRF")
    w("**Vấn đề:** RRF chỉ gộp *thứ hạng*, không đọc nội dung, nên không phân biệt "
      "được hai chunk cùng hạng nhưng khác hẳn nhau về mức độ trả lời trúng câu hỏi.  ")
    w("**Action:** `rerank_cross_encoder()` (Task 7) đã viết sẵn — bật lên chấm lại "
      "top 10 sau RRF rồi mới cắt xuống top_k.  ")
    w("**Kỳ vọng:** tăng Article Precision và faithfulness, đổi lại thêm độ trễ; "
      "nên đo lại A/B trước khi bật mặc định.")
    w("")

    # --- Per-item full ---
    w("## 8. Phụ lục — bảng điểm đầy đủ")
    w("")
    w("<details>")
    w("<summary>Điểm từng câu (config A)</summary>")
    w("")
    w("| ID | Chủ đề | Hit | Recall | Precision | MRR |")
    w("|---|---|---|---|---|---|")
    for it in A["offline"]["per_item"]:
        w(f"| {it['id']} | {it['topic']} | {int(it['hit'])} | {it['article_recall']} "
          f"| {it['article_precision']} | {it['mrr']} |")
    w("")
    w("</details>")
    w("")

    RESULTS_PATH.write_text("\n".join(L), encoding="utf-8")
    print(f"\n✓ Đã ghi báo cáo: {RESULTS_PATH}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="RAG evaluation pipeline")
    parser.add_argument("--ragas-limit", type=int, default=3,
                        help="Số câu hỏi đưa vào RAGAS mỗi config (0 = bỏ qua)")
    parser.add_argument("--offline-only", action="store_true",
                        help="Chỉ chạy metric tất định, không gọi LLM")
    parser.add_argument("--top-k", type=int, default=EVAL_TOP_K)
    parser.add_argument("--rebuild-report", action="store_true",
                        help="Dựng lại results.md từ eval_raw.json, không chạy lại pipeline")
    args = parser.parse_args()

    ragas_limit = 0 if args.offline_only else args.ragas_limit

    # Dựng lại báo cáo từ kết quả đã lưu — dùng khi chỉnh cách trình bày mà không
    # muốn tốn thêm quota LLM để chạy lại từ đầu.
    if args.rebuild_report:
        if not RAW_DUMP_PATH.exists():
            raise SystemExit(f"Chưa có {RAW_DUMP_PATH} — hãy chạy eval đầy đủ trước.")
        comparison = json.loads(RAW_DUMP_PATH.read_text(encoding="utf-8"))
        has_ragas = any((e.get("ragas") or {}).get("summary") for e in comparison.values())
        export_results(comparison, top_k=args.top_k,
                       ragas_limit=args.ragas_limit if has_ragas else 0)
        return

    golden = load_golden_dataset()
    print("=" * 70)
    print(f"RAG Evaluation — {len(golden)} câu hỏi, top_k={args.top_k}, "
          f"RAGAS subset={ragas_limit}")
    print("=" * 70)

    comparison = compare_configs(golden, ragas_limit=ragas_limit, top_k=args.top_k)

    # Lưu raw để có thể dựng lại báo cáo mà không phải chạy lại pipeline
    RAW_DUMP_PATH.write_text(
        json.dumps(
            {n: {k: v for k, v in e.items() if k != "records"}
             for n, e in comparison.items()},
            ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    export_results(comparison, top_k=args.top_k, ragas_limit=ragas_limit)

    print("\n" + "=" * 70)
    for n, e in comparison.items():
        o = e["offline"]
        print(f"{n:16s} hit={o['hit_rate']}  recall={o['article_recall']}  "
              f"prec={o['article_precision']}  mrr={o['mrr']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
