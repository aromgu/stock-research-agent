"""도구 호출 경로(trajectory) 평가 — 저장된 실행 기록만으로 계산하므로 API 호출이 없다.

"맞는 도구를 썼나"(인용 P/R)와 별개로 "어떻게 썼나"를 본다: 호출 횟수, 같은 호출 반복,
결과가 비어 헛수고가 된 호출. 에이전트 비용·지연의 원인이 어디서 오는지 보여준다.
"""

import json

# 도구가 "결과 없음"을 알릴 때 쓰는 문구 (tools.py의 반환 문자열 기준)
_EMPTY_MARKERS = ("찾지 못했습니다", "시세 데이터가 없습니다", "로컬 DB에 없음", "조회 불가:")


def _is_empty(result: str) -> bool:
    return any(m in result for m in _EMPTY_MARKERS)


def analyze_trajectory(run: dict) -> dict:
    """한 실행의 도구 호출 경로 지표: 호출 수, 중복 호출 수, 빈 결과 호출 수, LLM 스텝 수."""
    transcript = run.get("transcript")
    if transcript is None:
        # 고정 파이프라인은 코드가 정한 순서대로 도구마다 한 번씩만 부른다
        sections = run["context"].split("\n\n[") if run["predicted_sources"] else []
        return {
            "tool_calls": len(run["predicted_sources"]),
            "duplicate_calls": 0,
            "empty_calls": sum(_is_empty(s) for s in sections),
            "llm_steps": None,
        }

    seen = set()
    duplicates = 0
    for t in transcript:
        key = (t["tool"], json.dumps(t["args"], sort_keys=True, ensure_ascii=False))
        if key in seen:
            duplicates += 1
        seen.add(key)
    return {
        "tool_calls": len(transcript),
        "duplicate_calls": duplicates,
        "empty_calls": sum(_is_empty(t["result"]) for t in transcript),
        "llm_steps": max((t["step"] for t in transcript), default=0) + 1,  # 도구 호출 스텝들 + 최종 답변
    }
