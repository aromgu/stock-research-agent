"""Phase 5: 채점 파이프라인 + ablation 실험 러너.

Phase 3(고정 단일도구 baseline) vs Phase 4(플래너 에이전트)를 같은 골드 질문셋으로
돌려서 수치 정확도 / 인용 Precision-Recall을 비교한다 (Notion 계획서 §1.4).

실행: python -m src.eval.run_phase5
"""

import time
from datetime import datetime
from pathlib import Path

from ..agent import baseline
from ..agent.planner import run_agent
from .gold_set import GOLD_QUESTIONS
from .graders import accumulate_citation_counts, grade_numeric, is_deferred_answer, precision_recall_f1

LOG_PATH = Path(__file__).resolve().parent.parent.parent / "logs" / "phase5_eval.md"
SUMMARY_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "phase5_summary_export.md"

_TOOL_TO_SOURCE = {"get_financial_data": "dart", "search_news": "news", "get_stock_price": "price"}

APPROACHES = ["dart_baseline", "news_baseline", "all_tools_baseline", "agent"]
APPROACH_LABEL = {
    "dart_baseline": "Phase 3 · DART-only 고정 파이프라인",
    "news_baseline": "Phase 3 · News-only 고정 파이프라인",
    "all_tools_baseline": "Phase 3 · 항상 3도구 전부 고정 파이프라인",
    "agent": "Phase 4 · 플래너 에이전트",
}


def _run_dart_baseline(gq: dict) -> dict:
    start = time.perf_counter()
    answer = baseline.answer_with_dart(gq["question"], gq["company"])
    elapsed = time.perf_counter() - start
    return {"answer": answer, "predicted_sources": {"dart"}, "elapsed": elapsed, "transcript": None}


def _run_news_baseline(gq: dict) -> dict:
    start = time.perf_counter()
    answer = baseline.answer_with_news(gq["question"])
    elapsed = time.perf_counter() - start
    return {"answer": answer, "predicted_sources": {"news"}, "elapsed": elapsed, "transcript": None}


def _run_all_tools_baseline(gq: dict) -> dict:
    start = time.perf_counter()
    answer = baseline.answer_with_all_tools(gq["question"], gq["company"])
    elapsed = time.perf_counter() - start
    return {"answer": answer, "predicted_sources": {"dart", "news", "price"}, "elapsed": elapsed, "transcript": None}


def _run_agent(gq: dict) -> dict:
    start = time.perf_counter()
    result = run_agent(gq["question"])
    elapsed = time.perf_counter() - start
    predicted = {_TOOL_TO_SOURCE[t["tool"]] for t in result["transcript"] if t["tool"] in _TOOL_TO_SOURCE}
    return {
        "answer": result["answer"],
        "predicted_sources": predicted,
        "elapsed": elapsed,
        "transcript": result["transcript"],
    }


_RUNNERS = {
    "dart_baseline": _run_dart_baseline,
    "news_baseline": _run_news_baseline,
    "all_tools_baseline": _run_all_tools_baseline,
    "agent": _run_agent,
}


def _log_run(gq: dict, approach: str, run: dict, numeric_grade: dict) -> None:
    """실험 입출력을 logs/phase5_eval.md에 사람이 읽기 좋은 형태로 append."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts = [
        f"## [{timestamp}] {gq['id']} · {APPROACH_LABEL[approach]} · {run['elapsed']:.1f}초\n",
        f"### 질문\n{gq['question']}\n",
    ]
    if run["transcript"] is not None:
        for t in run["transcript"]:
            parts.append(f"### Step {t['step']}: `{t['tool']}({t['args']})`\n```\n{t['result']}\n```\n")
    parts.append(f"### 답변\n{run['answer']}\n")
    parts.append(
        f"### 채점\n"
        f"- 인용 출처(예측 / 정답): {sorted(run['predicted_sources'])} / {sorted(gq['expected_sources'])}\n"
        f"- 숫자 채점: {numeric_grade}\n"
        f"- 되묻고 회피했는지: {is_deferred_answer(run['answer'])}\n\n---\n"
    )
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main() -> None:
    citation_counts = {a: {"tp": 0, "fp": 0, "fn": 0} for a in APPROACHES}
    numeric_stats = {a: {"applicable": 0, "correct": 0} for a in APPROACHES}
    deferred_stats = {a: {"total": 0, "deferred": 0} for a in APPROACHES}
    rows = []  # 질문별 상세 표용

    for gq in GOLD_QUESTIONS:
        for approach in APPROACHES:
            run = _RUNNERS[approach](gq)
            numeric_grade = grade_numeric(run["answer"], gq["numeric_target"])
            deferred = is_deferred_answer(run["answer"])
            accumulate_citation_counts(run["predicted_sources"], gq["expected_sources"], citation_counts[approach])
            if numeric_grade["applicable"] and numeric_grade["correct"] is not None:
                numeric_stats[approach]["applicable"] += 1
                if numeric_grade["correct"]:
                    numeric_stats[approach]["correct"] += 1
            deferred_stats[approach]["total"] += 1
            if deferred:
                deferred_stats[approach]["deferred"] += 1
            _log_run(gq, approach, run, numeric_grade)
            rows.append(
                {
                    "question_id": gq["id"],
                    "approach": approach,
                    "predicted_sources": sorted(run["predicted_sources"]),
                    "expected_sources": sorted(gq["expected_sources"]),
                    "numeric_correct": numeric_grade["correct"],
                    "deferred": deferred,
                    "elapsed": run["elapsed"],
                }
            )
            print(f"[done] {gq['id']} / {approach} ({run['elapsed']:.1f}s)")

    _write_summary(rows, citation_counts, numeric_stats, deferred_stats)
    print(f"\n상세 로그: {LOG_PATH}")
    print(f"요약 리포트: {SUMMARY_PATH}")


def _write_summary(rows: list[dict], citation_counts: dict, numeric_stats: dict, deferred_stats: dict) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 5 채점 결과 (에이전틱 vs 고정 파이프라인 ablation)\n",
        f"생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · 질문 {len(GOLD_QUESTIONS)}개 x 방식 {len(APPROACHES)}개\n",
        "## 요약\n",
        "| 방식 | 인용 Precision | 인용 Recall | 인용 F1 | 수치 정확도 | 되묻고 회피 |",
        "|---|---|---|---|---|---|",
    ]
    for approach in APPROACHES:
        pr = precision_recall_f1(citation_counts[approach])
        ns = numeric_stats[approach]
        ds = deferred_stats[approach]
        acc = f"{ns['correct']}/{ns['applicable']}" if ns["applicable"] else "해당없음"
        p = f"{pr['precision']:.2f}" if pr["precision"] is not None else "N/A"
        r = f"{pr['recall']:.2f}" if pr["recall"] is not None else "N/A"
        f1 = f"{pr['f1']:.2f}" if pr["f1"] is not None else "N/A"
        lines.append(f"| {APPROACH_LABEL[approach]} | {p} | {r} | {f1} | {acc} | {ds['deferred']}/{ds['total']} |")

    lines.append("\n## 질문별 상세\n")
    lines.append("| 질문 | 방식 | 예측 출처 | 정답 출처 | 숫자 정답 | 되묻고 회피 | 소요(초) |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['question_id']} | {APPROACH_LABEL[r['approach']]} | {r['predicted_sources']} | "
            f"{r['expected_sources']} | {r['numeric_correct']} | {r['deferred']} | {r['elapsed']:.1f} |"
        )

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
