"""Phase 4: Planner (질문 분해, ReAct 루프로 도구 선택, 최종 답변 종합).

Thought -> Action(tool 선택) -> Observation을 LLM이 "충분한 정보가 모였다"고
판단할 때까지 반복 (Notion 계획서 §1.1). 무한 루프 방지로 최대 반복 횟수를 하드리밋.
"""

import json
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from .tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

MODEL = "gpt-5.4-nano"  # 저렴한 모델로 고정 (비용 최소화)
MAX_ITERATIONS = 7  # 계획서 §1.1: 무한 루프 방지 하드리밋 (원인 카테고리별 검색 여유를 위해 5→7)
LOG_PATH = Path(__file__).resolve().parent.parent.parent / "logs" / "agent_runs.md"

SYSTEM_PROMPT = (
    "너는 종목 리서치 어시스턴트야. get_financial_data(DART 재무 데이터)와 search_news(뉴스 검색) "
    "두 도구를 상황에 맞게 골라서 호출해 질문에 답해. 숫자가 필요하면 get_financial_data, "
    "실적 원인·전망·업계 이슈 같은 정성적 맥락이 필요하면 search_news를 써.\n\n"
    "'실적/가격이 왜 이랬는지' 같은 원인을 물으면:\n"
    "1. 원인은 하나가 아닐 수 있다는 전제로 최소한 "
    "아래 서로 다른 카테고리 중 관련 있어 보이는 걸 각각 별도 검색으로 확인해봐 — 한 카테고리에서 "
    "그럴듯한 답을 찾았다고 바로 멈추지 말고, 최소 2개 카테고리는 확인한 다음 답변을 작성해:\n"
    "  a) 회사 내부 이슈 (실적 발표 자체, 사업부별 실적, 경영/노사 이슈 등)\n"
    "  b) 산업/경쟁사 동향 (업황 사이클, 원자재·공급망, 경쟁사 실적)\n"
    "  c) 지정학적/거시경제 이슈 (전쟁, 수출규제, 관세, 환율 등 — 회사명이 기사에 안 나올 수도 있음)\n"
    "2. get_financial_data 결과에 나오는 '기간'(예: 2026-01-01~2026-06-30)을 확인하고, "
    "search_news를 부를 때 그 기간을 after/before로 지정해. 오늘 기준 최근 뉴스만 보면 "
    "그 분기 중간에 있었던 진짜 원인을 놓칠 수 있어.\n"
    "3. 검색어는 회사명+기간(1분기/반기/3분기/연간)에 '실적 발표', '컨퍼런스콜'처럼 그 "
    "시점 기사에 실제 쓰였을 이벤트 용어를 조합해서 만들어(예: '삼성전자 반기 실적 발표'). "
    "'실적', '영업이익' 같은 일반 단어만 쓰면 매일 나오는 전망성 기사에 밀려서 그 시점 기사를 "
    "못 찾을 수 있다. 회사마다/표현마다 결과가 다르니 한 번에 안 나오면 다른 표현으로 바꿔 "
    "재시도해봐.\n\n"
    "필요하면 여러 번 호출해도 되지만, 충분한 정보가 모이면 더 이상 도구를 호출하지 말고 "
    "어떤 데이터/기사를 근거로 했는지 밝히며 최종 답변을 작성해."
)

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def run_agent(question: str) -> dict:
    """ReAct 루프 실행. {"answer", "transcript"(스텝별 도구 호출 기록), "n_steps"} 반환."""
    client = _get_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    transcript = []
    start = time.perf_counter()
    n_steps = 0

    for step in range(1, MAX_ITERATIONS + 1):
        n_steps = step
        resp = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOL_SCHEMAS)
        msg = resp.choices[0].message

        if not msg.tool_calls:
            answer = msg.content
            break

        messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            }
        )
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            args = json.loads(tc.function.arguments)
            fn = TOOL_FUNCTIONS.get(fn_name)
            result = fn(**args) if fn else f"알 수 없는 도구: {fn_name}"
            transcript.append({"step": step, "tool": fn_name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    else:
        # 하드리밋 도달 - 도구 호출 없이 지금까지 모은 정보로 강제 답변 생성
        messages.append({"role": "user", "content": "지금까지 모은 정보로 최종 답변을 작성해줘 (더 이상 도구 호출 금지)."})
        resp = client.chat.completions.create(model=MODEL, messages=messages)
        answer = resp.choices[0].message.content

    elapsed = time.perf_counter() - start
    _log_run(question, transcript, answer, elapsed, n_steps)
    return {"answer": answer, "transcript": transcript, "n_steps": n_steps}


def _log_run(question: str, transcript: list[dict], answer: str, elapsed_sec: float, n_steps: int) -> None:
    """실행 기록을 logs/agent_runs.md에 사람이 읽기 좋은 형태로 append."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    parts = [f"## [{timestamp}] {n_steps}스텝 · {elapsed_sec:.1f}초 · {MODEL}\n", f"### 질문\n{question}\n"]
    for t in transcript:
        parts.append(f"### Step {t['step']}: `{t['tool']}({t['args']})`\n```\n{t['result']}\n```\n")
    parts.append(f"### 최종 답변\n{answer}\n\n---\n")

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(parts))
