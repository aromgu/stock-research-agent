"""Phase 4: Planner (질문 분해, ReAct 루프로 도구 선택, 최종 답변 종합).

Thought -> Action(tool 선택) -> Observation을 LLM이 "충분한 정보가 모였다"고
판단할 때까지 반복 (Notion 계획서 §1.1). 무한 루프 방지로 최대 반복 횟수를 하드리밋.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from .prompts import ANSWER_MODEL, ANSWER_RULE, PLANNER_MODEL, PREMISE_CHECK_RULE
from .tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

_MAX_PARALLEL_TOOLS = 4  # NAVER·DART·KRX에 한꺼번에 너무 많이 몰리지 않도록 동시 실행 수 제한
MAX_ITERATIONS = 7  # 계획서 §1.1: 무한 루프 방지 하드리밋 (원인 카테고리별 검색 여유를 위해 5→7)
LOG_PATH = Path(__file__).resolve().parent.parent.parent / "logs" / "agent_runs.md"

SYSTEM_PROMPT = (
    "너는 종목 리서치 어시스턴트야. get_financial_data(DART 재무 데이터), search_news(뉴스 검색), "
    "get_stock_price(주가·수익률·PER/PBR/시가총액), compare_peers(같은 업종 회사들과 재무 지표 순위 비교) "
    "네 도구와, 뉴스 기사 본문을 읽는 read_articles를 상황에 맞게 골라서 호출해 질문에 답해. "
    "재무 숫자가 필요하면 get_financial_data, 실적 원인·전망·업계 이슈 같은 정성적 맥락이 필요하면 "
    "search_news, 주가가 오르내린 사실이나 밸류에이션(비싼지/싼지) 확인이 필요하면 get_stock_price, "
    "'업종 1위', '경쟁사 대비', '같은 업종에서 높은 편인지' 같은 비교가 필요하면 compare_peers를 써. "
    "질문에 특정 연도(예: 2025년)가 있으면 get_financial_data의 year를 지정해.\n\n"
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
    "재시도해봐.\n"
    "4. search_news 결과는 제목과 한두 문장 요약뿐이다. 요약만으로 원인·수치가 분명하지 않을 때만 "
    "read_articles로 가장 관련 있는 기사 1~3개의 본문을 읽어 (결과의 id를 그대로 사용). 요약으로 충분하면 읽지 마.\n\n"
    "서로 독립적인 조회가 여러 개 필요하면(예: 두 회사의 재무, 카테고리별 뉴스 검색) 한 번의 응답에서 "
    "도구 호출을 여러 개 한꺼번에 요청해 — 동시에 실행되고, 단계를 나눌수록 느리고 비용이 커진다. "
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


def record_usage(usage: dict, resp, model: str) -> None:
    """LLM 호출 1회의 토큰 사용량을 모델별로 누적 (입력/출력/입력 중 캐시 적중)."""
    u = resp.usage
    if u is None:
        return
    details = getattr(u, "prompt_tokens_details", None)
    acc = usage.setdefault(model, {"calls": 0, "prompt": 0, "completion": 0, "cached": 0})
    acc["calls"] += 1
    acc["prompt"] += u.prompt_tokens or 0
    acc["completion"] += u.completion_tokens or 0
    acc["cached"] += (getattr(details, "cached_tokens", 0) or 0) if details else 0


def _run_tool(fn_name: str, args: dict) -> tuple[str, float]:
    fn = TOOL_FUNCTIONS.get(fn_name)
    start = time.perf_counter()
    try:
        result = fn(**args) if fn else f"알 수 없는 도구: {fn_name}"
    except Exception as e:  # noqa: BLE001 — 도구 오류로 에이전트 전체가 멈추지 않게, 오류를 결과로 돌려줘 재시도하게 함
        result = f"도구 실행 오류 ({type(e).__name__}): {e}"
    return result, time.perf_counter() - start


def _verify_answer(client: OpenAI, question: str, draft: str, transcript: list[dict], usage: dict) -> str:
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
        "- 질문의 전제(올랐다/줄었다 등)가 자료와 다르면 답변 첫머리에서 바로잡아. "
        "전제가 자료와 같으면 정정하는 표현('전제와 달리', '줄지 않고')을 쓰지 마.\n"
        "- 답변 안에서 결론끼리 모순되면 계산·자료 기준으로 하나로 맞춰.\n"
        "- 자료로 뒷받침되는 내용은 빼지 말고 유지해. 검토 과정 설명 없이 최종 답변만 출력해.\n\n"
        f"[질문]\n{question}\n\n[조회 자료]\n{evidence}\n\n[초안 답변]\n{draft}"
    )
    resp = client.chat.completions.create(model=ANSWER_MODEL, messages=[{"role": "user", "content": prompt}])
    record_usage(usage, resp, ANSWER_MODEL)
    return resp.choices[0].message.content or draft


def run_agent(question: str) -> dict:
    """ReAct 루프 실행 후 조회 자료로 답변을 검증한다.

    반환: {"answer"(검증 후), "draft_answer"(검증 전), "transcript"(도구 호출 기록, 호출별 소요 시간 포함),
    "n_steps", "financial_guard_triggered"(재무 도구를 안 불러 되돌렸는지), "usage"(모델별 토큰 사용량)}
    """
    client = _get_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    transcript = []
    usage: dict = {}
    start = time.perf_counter()
    n_steps = 0
    guard_triggered = False

    for step in range(1, MAX_ITERATIONS + 1):
        n_steps = step
        resp = client.chat.completions.create(model=PLANNER_MODEL, messages=messages, tools=TOOL_SCHEMAS)
        record_usage(usage, resp, PLANNER_MODEL)
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
        # 한 단계에서 요청한 도구들은 서로 독립이므로 동시에 실행한다 (두 회사 조회, 여러 뉴스 검색 등).
        # 결과는 요청한 순서대로 붙인다 — OpenAI는 tool_call_id로 짝을 맞추지만 로그를 읽기 쉽게.
        calls = [(tc, tc.function.name, json.loads(tc.function.arguments)) for tc in msg.tool_calls]
        with ThreadPoolExecutor(max_workers=min(len(calls), _MAX_PARALLEL_TOOLS)) as pool:
            outcomes = list(pool.map(lambda c: _run_tool(c[1], c[2]), calls))
        for (tc, fn_name, args), (result, tool_elapsed) in zip(calls, outcomes):
            transcript.append(
                {"step": step, "tool": fn_name, "args": args, "result": result, "elapsed": round(tool_elapsed, 2)}
            )
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    else:
        # 하드리밋 도달 - 도구 호출 없이 지금까지 모은 정보로 강제 답변 생성
        messages.append({"role": "user", "content": "지금까지 모은 정보로 최종 답변을 작성해줘 (더 이상 도구 호출 금지)."})
        resp = client.chat.completions.create(model=PLANNER_MODEL, messages=messages)
        record_usage(usage, resp, PLANNER_MODEL)
        answer = resp.choices[0].message.content

    draft = answer
    answer = _verify_answer(client, question, draft, transcript, usage)
    elapsed = time.perf_counter() - start
    _log_run(question, transcript, answer, elapsed, n_steps)
    return {
        "answer": answer,
        "draft_answer": draft,
        "transcript": transcript,
        "n_steps": n_steps,
        "financial_guard_triggered": guard_triggered,
        "usage": usage,
    }


def _log_run(question: str, transcript: list[dict], answer: str, elapsed_sec: float, n_steps: int) -> None:
    """실행 기록을 logs/agent_runs.md에 사람이 읽기 좋은 형태로 append."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    parts = [f"## [{timestamp}] {n_steps}스텝 · {elapsed_sec:.1f}초 · 계획 {PLANNER_MODEL} / 답변 {ANSWER_MODEL}\n", f"### 질문\n{question}\n"]
    for t in transcript:
        parts.append(f"### Step {t['step']}: `{t['tool']}({t['args']})`\n```\n{t['result']}\n```\n")
    parts.append(f"### 최종 답변\n{answer}\n\n---\n")

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(parts))
