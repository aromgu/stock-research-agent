"""Phase 4: Planner (질문 분해, ReAct 루프로 도구 선택, 최종 답변 종합).

Thought -> Action(tool 선택) -> Observation을 LLM이 "충분한 정보가 모였다"고
판단할 때까지 반복 (Notion 계획서 §1.1). 무한 루프 방지로 최대 반복 횟수를 하드리밋.
"""

import json
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from .prompts import ANSWER_RULE, PREMISE_CHECK_RULE
from .tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

MODEL = "gpt-5.4-nano"  # 저렴한 모델로 고정 (비용 최소화)
MAX_ITERATIONS = 7  # 계획서 §1.1: 무한 루프 방지 하드리밋 (원인 카테고리별 검색 여유를 위해 5→7)
LOG_PATH = Path(__file__).resolve().parent.parent.parent / "logs" / "agent_runs.md"

SYSTEM_PROMPT = (
    "너는 종목 리서치 어시스턴트야. get_financial_data(DART 재무 데이터), search_news(뉴스 검색), "
    "get_stock_price(주가·수익률·PER/PBR/시가총액) 세 도구를 상황에 맞게 골라서 호출해 질문에 답해. "
    "재무 숫자가 필요하면 get_financial_data, 실적 원인·전망·업계 이슈 같은 정성적 맥락이 필요하면 "
    "search_news, 주가가 오르내린 사실이나 밸류에이션(비싼지/싼지) 확인이 필요하면 get_stock_price를 "
    "써.\n\n"
    f"{PREMISE_CHECK_RULE}\n\n"
    f"{ANSWER_RULE}\n\n"
    "'실적/가격이 왜 이랬는지' 같은 원인을 물으면:\n"
    "1. 원인은 하나가 아닐 수 있다는 전제로 최소한 "
    "아래 서로 다른 카테고리 중 관련 있어 보이는 걸 각각 별도 검색으로 확인해봐 — 한 카테고리에서 "
    "그럴듯한 답을 찾았다고 바로 멈추지 말고, 최소 2개 카테고리는 확인한 다음 답변을 작성해:\n"
    "  a) 회사 내부 이슈 (실적 발표 자체, 사업부별 실적, 경영/노사 이슈 등)\n"
    "  b) 산업/경쟁사 동향 (업황 사이클, 원자재·공급망, 경쟁사 실적)\n"
    "  c) 지정학적/거시경제 이슈 (전쟁, 수출규제, 관세, 환율 등 — 회사명이 기사에 안 나올 수도 있음)\n"
    "  질문이 카테고리를 직접 지정하면(예: '내부/산업/거시로 나눠서'), 지정된 카테고리마다 최소 한 번씩 "
    "그 카테고리에 맞는 검색어로 따로 검색해. 거시경제는 회사명을 빼고 '2026년 상반기 환율 반도체 수출'처럼 "
    "검색해야 기사가 나오는 경우가 많다.\n"
    "2. get_financial_data 결과에 나오는 '기간'(예: 2026-01-01~2026-06-30)을 확인하고, "
    "search_news를 부를 때 after는 그 기간 시작일, before는 기간 종료일로부터 약 2개월 뒤"
    "(예: 2026-08-31)로 지정해. 실적의 원인 설명은 대부분 기간이 끝난 뒤 실적 발표·컨퍼런스콜 "
    "때 기사로 나오므로, 기간 종료 후 두 달 사이에 나온 실적 발표 기사는 그 기간의 원인을 "
    "설명하는 정당한 근거다. 오늘 기준 최근 뉴스만 보면 그 기간의 진짜 원인을 놓칠 수 있어.\n"
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


# 재무 수치·실적 방향을 다루는 질문인데 재무 도구를 한 번도 안 부르고 답하면, 모델이 수치를 지어낸다
# (전년 영업이익을 115.8조로 지어내 "줄었다"는 틀린 전제를 받아들인 사례, 실제 16.7조). 코드로 막는다.
_FINANCIAL_TERMS = ("실적", "매출", "영업이익", "순이익", "이익률", "부채", "자산", "자본", "재무", "흑자", "적자")
_FINANCIAL_GUARD_MESSAGE = (
    "이 질문은 재무 수치나 실적을 다루는데 아직 get_financial_data를 호출하지 않았어. "
    "수치를 추측하지 말고 get_financial_data로 확인한 다음 다시 답해."
)


def _needs_financial_data(question: str, transcript: list[dict]) -> bool:
    asks_financials = any(term in question for term in _FINANCIAL_TERMS)
    return asks_financials and not any(t["tool"] == "get_financial_data" for t in transcript)


def _verify_answer(client: OpenAI, question: str, draft: str, transcript: list[dict]) -> str:
    """초안을 조회 자료와 대조해 고친다: 자료와 다른 수치, 근거 없는 주장, 앞뒤 모순.

    nano는 자료를 찾아와도 자료에 없는 원인을 끼워 넣거나(예: '성과급'), 결론을 앞뒤로 다르게
    쓰는 일이 잦았다. 답변을 쓴 뒤 자료만 보고 한 번 더 검토하는 단계를 둔다 (nano 1회 추가).
    """
    if not transcript:
        return draft  # 조회 자료가 없으면 대조할 것이 없음
    evidence = "\n\n".join(f"[{t['tool']}({t['args']})]\n{t['result']}" for t in transcript)
    prompt = (
        "아래 [초안 답변]을 [조회 자료]와 대조해서 고친 최종 답변을 작성해.\n"
        "- 수치는 조회 자료의 값과 같아야 한다. 다르면 자료의 값으로 고쳐.\n"
        "- 원인이나 사실 주장 중 조회 자료로 뒷받침되지 않는 것은 삭제하거나 '추정'이라고 분명히 표시해.\n"
        "- 질문의 전제(올랐다/줄었다 등)가 자료와 다르면 답변 첫머리에서 바로잡아.\n"
        "- 답변 안에서 결론끼리 모순되면 계산·자료 기준으로 하나로 맞춰.\n"
        "- 자료로 뒷받침되는 내용은 빼지 말고 유지해. 검토 과정 설명 없이 최종 답변만 출력해.\n\n"
        f"[질문]\n{question}\n\n[조회 자료]\n{evidence}\n\n[초안 답변]\n{draft}"
    )
    resp = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content or draft


def run_agent(question: str) -> dict:
    """ReAct 루프 실행 후 조회 자료로 답변을 검증한다.

    반환: {"answer"(검증 후), "draft_answer"(검증 전), "transcript"(도구 호출 기록), "n_steps",
    "financial_guard_triggered"(재무 도구를 안 불러 되돌렸는지)}
    """
    client = _get_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    transcript = []
    start = time.perf_counter()
    n_steps = 0
    guard_triggered = False

    for step in range(1, MAX_ITERATIONS + 1):
        n_steps = step
        resp = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOL_SCHEMAS)
        msg = resp.choices[0].message

        if not msg.tool_calls:
            if not guard_triggered and _needs_financial_data(question, transcript):
                guard_triggered = True  # 한 번만 되돌린다 (무한 반복 방지)
                messages.append({"role": "assistant", "content": msg.content})
                messages.append({"role": "user", "content": _FINANCIAL_GUARD_MESSAGE})
                continue
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

    draft = answer
    answer = _verify_answer(client, question, draft, transcript)
    elapsed = time.perf_counter() - start
    _log_run(question, transcript, answer, elapsed, n_steps)
    return {
        "answer": answer,
        "draft_answer": draft,
        "transcript": transcript,
        "n_steps": n_steps,
        "financial_guard_triggered": guard_triggered,
    }


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
