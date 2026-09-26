"""도구 계층 벤치마크: 저장된 에이전트 실행 기록의 도구 호출을 OpenAI 없이 다시 실행해 지연 시간과
외부 API 호출 수를 잰다. 캐시·아카이브 같은 도구 계층 개선의 효과를 LLM 비용 없이 확인하는 용도.

실행: python -m src.eval.bench_tools [--runs data/phase5_runs_export.jsonl] [--limit 40]

1회차: 비어 있는 별도 캐시 파일로 시작 (처음 묻는 상황)
2회차: 같은 캐시로 한 번 더 (같은 질문을 다시 묻거나 평가를 재실행하는 상황)
임베딩 모델 로드 시간(약 13초)은 두 회차에 공정하도록 측정 전에 미리 끝낸다.
"""

import argparse
import json
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from ..agent import tools
from ..data import cache, news_client

_USAGE_FILE = Path(news_client.__file__).parent / ".news_api_usage.json"


def _naver_count() -> int:
    if not _USAGE_FILE.exists():
        return 0
    return sum(json.loads(_USAGE_FILE.read_text(encoding="utf-8")).values())


def _load_calls(runs_path: Path, limit: int) -> list[tuple[str, dict]]:
    calls = []
    for line in runs_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["approach"] == "agent" and record.get("transcript"):
            calls += [(t["tool"], t["args"]) for t in record["transcript"]]
    return calls[:limit]


def _run_pass(calls: list[tuple[str, dict]]) -> dict:
    per_tool = defaultdict(lambda: {"n": 0, "sec": 0.0})
    naver_before = _naver_count()
    start = time.perf_counter()
    for name, args in calls:
        t = time.perf_counter()
        try:
            tools.TOOL_FUNCTIONS[name](**args)
        except Exception as e:  # noqa: BLE001 — 한 호출이 실패해도 벤치마크는 계속
            print(f"  [error] {name}{args}: {e}")
        per_tool[name]["n"] += 1
        per_tool[name]["sec"] += time.perf_counter() - t
    return {"total": time.perf_counter() - start, "naver": _naver_count() - naver_before, "per_tool": dict(per_tool)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=str(Path(__file__).resolve().parent.parent.parent / "data" / "phase5_runs_export.jsonl"))
    parser.add_argument("--limit", type=int, default=40, help="다시 실행할 도구 호출 수 (뉴스 기간 검색은 호출당 수십 초)")
    args = parser.parse_args()

    calls = _load_calls(Path(args.runs), args.limit)
    print(f"도구 호출 {len(calls)}개 재실행 (저장된 에이전트 기록 기준)")
    news_client.embed_texts(["모델 미리 로드"])  # 모델 로드 시간은 측정에서 제외

    real_cache = cache.CACHE_PATH
    with tempfile.TemporaryDirectory() as tmp:
        cache.CACHE_PATH = Path(tmp) / "bench_cache.db"  # 실제 캐시를 건드리지 않는 빈 캐시
        try:
            first = _run_pass(calls)
            second = _run_pass(calls)
        finally:
            cache.CACHE_PATH = real_cache

    print("\n| 도구 | 호출 수 | 1회차(초) | 2회차(초) |\n|---|---|---|---|")
    for name in first["per_tool"]:
        a, b = first["per_tool"][name], second["per_tool"][name]
        print(f"| {name} | {a['n']} | {a['sec']:.1f} | {b['sec']:.1f} |")
    print(f"| **합계** | {len(calls)} | **{first['total']:.1f}** | **{second['total']:.1f}** |")
    print(f"\nNAVER API 호출: 1회차 {first['naver']}번 → 2회차 {second['naver']}번")


if __name__ == "__main__":
    main()
