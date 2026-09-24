"""Phase 5: 채점 파이프라인 + ablation 실험 러너.

Phase 3(고정 파이프라인 baseline) vs Phase 4(플래너 에이전트)를 같은 골드 질문셋으로
돌려서 수치 정확도 / 인용 Precision-Recall / LLM 심판 / 쌍대 비교 / 도구 호출 경로를 비교한다
(Notion 계획서 §1.4). 결과는 튜닝 셋과 held-out 셋을 나눠 보고한다 (gold_set.py 참고).

실행:
  python -m src.eval.run_phase5               답변 새로 생성(저장) + 채점
  python -m src.eval.run_phase5 --resume      끊긴 생성을 이어서 (저장된 건 건너뜀) + 채점
  python -m src.eval.run_phase5 --regrade     생성 없이 저장된 답변으로 채점만 (채점 로직을 고쳤을 때)
  python -m src.eval.run_phase5 --skip-judge  LLM 심판 없이 (OpenAI 호출 없이 흐름 점검)
  --questions / --approaches 로 일부만 실행 가능 (이때 요약 리포트는 갱신 안 함)

답변 생성(에이전트 루프 최대 7회 호출)이 API 비용의 대부분이라, 생성 결과를
data/phase5_runs_export.jsonl에 저장해두고 채점기만 바뀌면 --regrade로 재사용한다.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from ..agent import baseline
from ..agent.planner import run_agent
from .gold_set import GOLD_QUESTIONS
from .graders import (
    accumulate_citation_counts,
    describe_numeric_check,
    describe_reference,
    grade_numeric,
    is_deferred_answer,
    judge_answer,
    pairwise_judge,
    precision_recall_f1,
)
from .trajectory import analyze_trajectory

LOG_PATH = Path(__file__).resolve().parent.parent.parent / "logs" / "phase5_eval.md"
SUMMARY_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "phase5_summary_export.md"
RUNS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "phase5_runs_export.jsonl"

_TOOL_TO_SOURCE = {"get_financial_data": "dart", "search_news": "news", "get_stock_price": "price"}

APPROACHES = ["dart_baseline", "news_baseline", "all_tools_baseline", "agent"]
APPROACH_LABEL = {
    "dart_baseline": "Phase 3 · DART-only 고정 파이프라인",
    "news_baseline": "Phase 3 · News-only 고정 파이프라인",
    "all_tools_baseline": "Phase 3 · 항상 3도구 전부 고정 파이프라인",
    "agent": "Phase 4 · 플래너 에이전트",
}
# 쌍대 비교는 에이전트 vs 가장 강한 비교군 하나만 (심판 호출 = 질문 수 × 2회, 순서 바꿔 두 번)
PAIRWISE_PAIR = ("agent", "all_tools_baseline")

# 요약 표에 보여줄 순서 (주 지표가 먼저)
_SPLITS = (("heldout_v2", "held-out v2"), ("heldout", "held-out v1"), ("tuning", "튜닝 셋"))
_SPLIT_NOTE = {
    "heldout_v2": "결과를 보기 전에 확정, 현재 주 지표",
    "heldout": "h5를 보고 에이전트·채점을 고쳐 일부 오염됨",
    "tuning": "이 질문들을 보고 프롬프트·채점기를 고쳤음 — 과대평가 가능",
}


def _run_dart_baseline(gq: dict) -> dict:
    start = time.perf_counter()
    result = baseline.answer_with_dart(gq["question"], gq["company"])
    elapsed = time.perf_counter() - start
    return {**result, "predicted_sources": {"dart"}, "elapsed": elapsed, "transcript": None}


def _run_news_baseline(gq: dict) -> dict:
    start = time.perf_counter()
    result = baseline.answer_with_news(gq["question"])
    elapsed = time.perf_counter() - start
    return {**result, "predicted_sources": {"news"}, "elapsed": elapsed, "transcript": None}


def _run_all_tools_baseline(gq: dict) -> dict:
    start = time.perf_counter()
    result = baseline.answer_with_all_tools(gq["question"], gq["company"])
    elapsed = time.perf_counter() - start
    return {**result, "predicted_sources": {"dart", "news", "price"}, "elapsed": elapsed, "transcript": None}


def _run_agent(gq: dict) -> dict:
    start = time.perf_counter()
    result = run_agent(gq["question"])
    elapsed = time.perf_counter() - start
    predicted = {_TOOL_TO_SOURCE[t["tool"]] for t in result["transcript"] if t["tool"] in _TOOL_TO_SOURCE}
    context = "\n\n".join(f"[{t['tool']}({t['args']})]\n{t['result']}" for t in result["transcript"])
    return {
        "answer": result["answer"],
        "draft_answer": result["draft_answer"],
        "financial_guard_triggered": result["financial_guard_triggered"],
        "context": context,
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


def _log_run(gq: dict, approach: str, run: dict, numeric_grade: dict, reference: str | None, judge: dict) -> None:
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
    else:
        parts.append(f"### 입력 자료 (고정 파이프라인)\n```\n{run['context']}\n```\n")
    if run.get("draft_answer") and run["draft_answer"] != run["answer"]:
        guard = " (재무 도구를 안 불러 한 번 되돌림)" if run.get("financial_guard_triggered") else ""
        parts.append(f"### 초안 (검증 단계 전){guard}\n{run['draft_answer']}\n")
    parts.append(f"### 답변\n{run['answer']}\n")
    parts.append(
        f"### 채점\n"
        f"- 인용 출처(예측 / 정답): {sorted(run['predicted_sources'])} / {sorted(gq['expected_sources'])}\n"
        f"- 숫자 채점: {numeric_grade}\n"
        f"- 되묻고 회피했는지: {is_deferred_answer(run['answer'])}\n"
        f"- 도구 호출 경로: {analyze_trajectory(run)}\n"
        f"### LLM 심판\n"
        f"- 채점 기준: {gq['rubric']}\n"
        f"- 참고 정답: {reference or '없음'}\n"
        f"- 판정: {'PASS' if judge['pass'] else 'FAIL' if judge['pass'] is False else '판정 불가'}\n"
        f"- 근거: {judge['reason']}\n\n---\n"
    )
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(parts))


def _log_pairwise(gq: dict, result: dict) -> None:
    a, b = PAIRWISE_PAIR
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(
            f"## [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {gq['id']} · 쌍대 비교 "
            f"({APPROACH_LABEL[a]} vs {APPROACH_LABEL[b]})\n\n"
            f"- 최종 승자: {result['winner']} (순서 바꿔도 일관: {result['consistent']})\n"
            f"- 순서 1 (A={a}) 판정: {result['order1']['pick']} — {result['order1']['reason']}\n"
            f"- 순서 2 (A={b}) 판정: {result['order2']['pick']} — {result['order2']['reason']}\n\n---\n"
        )


def _load_runs() -> dict[tuple[str, str], dict]:
    if not RUNS_PATH.exists():
        return {}
    with open(RUNS_PATH, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return {(r["question_id"], r["approach"]): r for r in records}


def _save_runs(runs: dict[tuple[str, str], dict]) -> None:
    RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RUNS_PATH, "w", encoding="utf-8") as f:
        for record in runs.values():
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _generate(questions: list[dict], approaches: list[str], resume: bool) -> None:
    """답변 생성 단계 (API 비용 대부분). 선택한 (질문, 방식)만 새로 생성해 저장된 답변에서 교체한다.

    한 건 끝날 때마다 바로 저장해서 중간에 끊겨도 --resume으로 이어갈 수 있다.
    """
    runs = _load_runs()
    for gq in questions:
        for approach in approaches:
            if resume and (gq["id"], approach) in runs:
                continue
            run = _RUNNERS[approach](gq)
            runs[(gq["id"], approach)] = {
                "question_id": gq["id"],
                "approach": approach,
                "answer": run["answer"],
                "context": run["context"],
                "predicted_sources": sorted(run["predicted_sources"]),
                "elapsed": run["elapsed"],
                "transcript": run["transcript"],
                # 에이전트만 있는 필드 (검증 단계 전 초안, 재무 도구 강제 여부)
                "draft_answer": run.get("draft_answer"),
                "financial_guard_triggered": run.get("financial_guard_triggered"),
            }
            _save_runs(runs)
            print(f"[generated] {gq['id']} / {approach} ({run['elapsed']:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="저장된 답변은 건너뛰고 나머지만 생성")
    mode.add_argument("--regrade", action="store_true", help="생성 없이 저장된 답변으로 채점만")
    parser.add_argument("--questions", help="쉼표로 구분한 질문 id만 실행 (예: q5_multihop_samsung,h1_debt_ratio)")
    parser.add_argument("--approaches", help=f"쉼표로 구분한 방식만 실행 (선택: {','.join(APPROACHES)})")
    parser.add_argument("--skip-judge", action="store_true", help="LLM 심판·쌍대 비교 생략 (OpenAI 호출 없이 흐름 점검)")
    args = parser.parse_args()

    questions = GOLD_QUESTIONS
    if args.questions:
        wanted = set(args.questions.split(","))
        questions = [gq for gq in GOLD_QUESTIONS if gq["id"] in wanted]
        unknown = wanted - {gq["id"] for gq in questions}
        if unknown:
            parser.error(f"알 수 없는 질문 id: {sorted(unknown)}")
    approaches = args.approaches.split(",") if args.approaches else APPROACHES
    if set(approaches) - set(APPROACHES):
        parser.error(f"알 수 없는 방식: {sorted(set(approaches) - set(APPROACHES))}")
    partial = questions is not GOLD_QUESTIONS or approaches is not APPROACHES

    if not args.regrade:
        _generate(questions, approaches, resume=args.resume)
    runs = _load_runs()

    rows = []  # 한 줄 = (질문, 방식) 하나의 채점 결과. 요약 표는 이걸 split별로 집계
    pairwise_rows = []
    for gq in questions:
        reference = describe_reference(gq["numeric_target"])  # 질문당 한 번만 계산 (방식 무관)
        numeric_checks = {}
        for approach in approaches:
            run = runs.get((gq["id"], approach))
            if run is None:
                print(f"[skip] {gq['id']} / {approach}: 저장된 답변 없음 (--resume으로 생성 필요)")
                continue
            numeric_grade = grade_numeric(run["answer"], gq["numeric_target"])
            numeric_checks[approach] = describe_numeric_check(numeric_grade)
            if args.skip_judge:
                judge = {"pass": None, "reason": "--skip-judge로 생략"}
            else:
                judge = judge_answer(
                    gq["question"], gq["rubric"], run["answer"], reference, numeric_checks[approach], run["context"]
                )
            _log_run(gq, approach, run, numeric_grade, reference, judge)
            rows.append(
                {
                    "question_id": gq["id"],
                    "split": gq["split"],
                    "approach": approach,
                    "predicted_sources": run["predicted_sources"],
                    "expected_sources": sorted(gq["expected_sources"]),
                    "numeric_correct": numeric_grade["correct"],
                    "deferred": is_deferred_answer(run["answer"]),
                    "judge_pass": judge["pass"],
                    "elapsed": run["elapsed"],
                    **analyze_trajectory(run),
                }
            )
            print(f"[graded] {gq['id']} / {approach} judge={judge['pass']}")

        a, b = PAIRWISE_PAIR
        if not args.skip_judge and a in numeric_checks and b in numeric_checks:
            result = pairwise_judge(
                gq["question"],
                gq["rubric"],
                reference,
                {"name": a, "answer": runs[(gq["id"], a)]["answer"], "numeric_check": numeric_checks[a]},
                {"name": b, "answer": runs[(gq["id"], b)]["answer"], "numeric_check": numeric_checks[b]},
            )
            _log_pairwise(gq, result)
            pairwise_rows.append({"question_id": gq["id"], "split": gq["split"], **result})
            print(f"[pairwise] {gq['id']} winner={result['winner']} consistent={result['consistent']}")

    print(f"\n상세 로그: {LOG_PATH}")
    if partial:
        # 일부만 채점한 결과로 전체 요약 표를 덮어쓰면 오해를 부르므로 갱신하지 않는다
        print("일부 질문/방식만 실행해서 요약 리포트는 갱신하지 않았습니다.")
        return
    _write_summary(rows, pairwise_rows)
    print(f"요약 리포트: {SUMMARY_PATH}")


def _fmt(v: float | None) -> str:
    return f"{v:.2f}" if v is not None else "N/A"


def _summary_table(rows: list[dict]) -> list[str]:
    lines = [
        "| 방식 | 인용 P | 인용 R | 인용 F1 | 수치 정확도 | 되묻고 회피 | LLM 심판 통과 |",
        "|---|---|---|---|---|---|---|",
    ]
    for approach in APPROACHES:
        rs = [r for r in rows if r["approach"] == approach]
        counts = {"tp": 0, "fp": 0, "fn": 0}
        for r in rs:
            accumulate_citation_counts(set(r["predicted_sources"]), set(r["expected_sources"]), counts)
        pr = precision_recall_f1(counts)
        numeric = [r["numeric_correct"] for r in rs if r["numeric_correct"] is not None]
        judged = [r["judge_pass"] for r in rs if r["judge_pass"] is not None]
        lines.append(
            f"| {APPROACH_LABEL[approach]} | {_fmt(pr['precision'])} | {_fmt(pr['recall'])} | {_fmt(pr['f1'])} "
            f"| {sum(numeric)}/{len(numeric)} | {sum(r['deferred'] for r in rs)}/{len(rs)} "
            f"| {sum(judged)}/{len(judged)} |"
        )
    return lines


def _trajectory_table(rows: list[dict]) -> list[str]:
    lines = [
        "| 방식 | 평균 도구 호출 | 중복 호출 | 빈 결과 호출 | 평균 LLM 스텝 | 평균 소요(초) |",
        "|---|---|---|---|---|---|",
    ]
    for approach in APPROACHES:
        rs = [r for r in rows if r["approach"] == approach]
        if not rs:
            continue
        calls = sum(r["tool_calls"] for r in rs)
        steps = [r["llm_steps"] for r in rs if r["llm_steps"] is not None]
        avg_steps = f"{sum(steps) / len(steps):.1f}" if steps else "-"
        lines.append(
            f"| {APPROACH_LABEL[approach]} | {calls / len(rs):.1f} | {sum(r['duplicate_calls'] for r in rs)} "
            f"| {sum(r['empty_calls'] for r in rs)}/{calls} | {avg_steps} "
            f"| {sum(r['elapsed'] for r in rs) / len(rs):.1f} |"
        )
    return lines


def _pairwise_table(pairwise_rows: list[dict]) -> list[str]:
    a, b = PAIRWISE_PAIR
    lines = [
        f"| 구분 | {APPROACH_LABEL[a]} 승 | {APPROACH_LABEL[b]} 승 | 무승부 | 순서 바꿔도 일관된 판정 |",
        "|---|---|---|---|---|",
    ]
    for split_name, label in (*_SPLITS, (None, "전체")):
        ps = [p for p in pairwise_rows if split_name is None or p["split"] == split_name]
        if not ps:
            continue
        lines.append(
            f"| {label} | {sum(p['winner'] == a for p in ps)} | {sum(p['winner'] == b for p in ps)} "
            f"| {sum(p['winner'] == 'tie' for p in ps)} | {sum(p['consistent'] for p in ps)}/{len(ps)} |"
        )
    return lines


def _write_summary(rows: list[dict], pairwise_rows: list[dict]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    counts = " + ".join(f"{label} {sum(q['split'] == s for q in GOLD_QUESTIONS)}" for s, label in _SPLITS)
    a, b = PAIRWISE_PAIR
    split_sections = []
    for split_name, label in _SPLITS:
        split_sections += [f"\n## {label} — {_SPLIT_NOTE[split_name]}\n", *_summary_table([r for r in rows if r["split"] == split_name])]
    lines = [
        "# Phase 5 채점 결과 (에이전틱 vs 고정 파이프라인 ablation)\n",
        f"생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · 질문 {len(GOLD_QUESTIONS)}개 "
        f"({counts}) x 방식 {len(APPROACHES)}개\n",
        *split_sections,
        f"\n## 쌍대 비교 ({APPROACH_LABEL[a]} vs {APPROACH_LABEL[b]}, 순서를 바꿔 두 번 판정하고 엇갈리면 무승부)\n",
        *_pairwise_table(pairwise_rows),
        "\n## 도구 호출 경로 (전체 질문)\n",
        *_trajectory_table(rows),
        "\n## 질문별 상세\n",
        "| 질문 | 구분 | 방식 | 예측 출처 | 정답 출처 | 숫자 정답 | 심판 | 도구 호출 | 소요(초) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['question_id']} | {r['split']} | {APPROACH_LABEL[r['approach']]} | {r['predicted_sources']} | "
            f"{r['expected_sources']} | {r['numeric_correct']} | {r['judge_pass']} | {r['tool_calls']} "
            f"| {r['elapsed']:.1f} |"
        )
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
