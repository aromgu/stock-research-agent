"""Phase 3: 고정 파이프라인 베이스라인 (DART만 / News만 / 항상 모든 도구).

Phase 4의 멀티홉 에이전트(플래너+동적 도구선택) 전 단계로, 도구 선택 능력이 없는
가장 단순한 RAG의 성능을 잰다 (Notion 계획서 §1.4 ablation 비교용).

공정한 비교를 위해 도구 자체는 `tools.py`(Phase 4 에이전트와 동일)를 그대로 불러 쓴다 —
baseline과 에이전트가 다른 건 "어떤 도구를 언제 쓰는지"뿐이어야 하기 때문.
"""

import time
from datetime import date, datetime, timedelta
from pathlib import Path

from openai import OpenAI

from . import tools
from .planner import record_usage
from .prompts import ANSWER_MODEL, ANSWER_RULE, PLANNER_MODEL, PREMISE_CHECK_RULE

LOG_PATH = Path(__file__).resolve().parent.parent.parent / "logs" / "baseline_experiments.md"


def _log_experiment(tool: str, question: str, prompt: str, answer: str, elapsed_sec: float) -> None:
    """실험 입출력을 logs/baseline_experiments.md에 사람이 읽기 좋은 형태로 append."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = (
        f"## [{timestamp}] {tool.upper()} · 답변 {ANSWER_MODEL} · {elapsed_sec:.1f}초\n\n"
        f"### 질문\n{question}\n\n"
        f"### 입력 (LLM에 보낸 프롬프트)\n```\n{prompt}\n```\n\n"
        f"### 출력\n{answer}\n\n"
        f"---\n\n"
    )
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(entry)


_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _ask(prompt: str, usage: dict) -> str:
    # 에이전트(planner.py)와 같은 조건이 되도록 출력 길이 제한을 두지 않는다 — 예전 300토큰
    # 제한 때문에 baseline 답변이 수치를 말하기도 전에 잘려서 불공정하게 졌다.
    resp = _get_client().chat.completions.create(
        model=ANSWER_MODEL,  # 에이전트의 최종 답변(검증 단계)과 같은 모델
        messages=[{"role": "user", "content": prompt}],
    )
    record_usage(usage, resp, ANSWER_MODEL)
    return resp.choices[0].message.content


def answer_with_dart(question: str, name_or_stock_code: str) -> dict:
    """DART 데이터만 근거로 질문에 답한다 (뉴스/주가 없이). {"answer", "context", "usage"} 반환."""
    usage: dict = {}
    start = time.perf_counter()
    context = tools.get_financial_data(name_or_stock_code)
    prompt = (
        f"다음은 DART 공시 재무 데이터다:\n{context}\n\n"
        f"{PREMISE_CHECK_RULE}\n{ANSWER_RULE}\n\n"
        f"이 데이터만 근거로 질문에 답해줘. 데이터에 없는 내용(예: 주가, 뉴스 맥락)은 "
        f"모른다고 답해. 질문: {question}"
    )
    answer = _ask(prompt, usage)
    elapsed = time.perf_counter() - start
    _log_experiment("dart", question, prompt, answer, elapsed)
    return {"answer": answer, "context": context, "usage": usage}


def _build_search_query(question: str, usage: dict) -> str:
    """질문 문장을 검색엔진에 적합한 짧은 검색어로 바꾸는 단발성 LLM 호출.

    Phase 4 에이전트와 달리 재시도/재구성 없이 딱 한 번만 변환한다 — "단일 도구
    baseline"이 순수 질문 문장을 그대로 검색어로 써서(구현 누락) 불공평하게 지는
    걸 막기 위한 최소 보정. 도구를 동적으로 여러 번 쓰는 능력 자체는 여전히 없다.
    """
    prompt = (
        "다음 질문을 뉴스 검색엔진에 넣을 짧은 검색어로 바꿔줘. "
        "회사명과 핵심 키워드만 남기고, 다른 설명 없이 검색어만 출력해.\n\n"
        f"질문: {question}"
    )
    resp = _get_client().chat.completions.create(
        model=PLANNER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=50,
    )
    record_usage(usage, resp, PLANNER_MODEL)
    return resp.choices[0].message.content.strip()


def answer_with_news(question: str) -> dict:
    """뉴스 검색 결과만 근거로 질문에 답한다 (DART/주가 없이). {"answer", "context", "usage"} 반환."""
    usage: dict = {}
    start = time.perf_counter()  # 검색어 생성부터 LLM 최종 답변까지 전체 소요시간
    search_query = _build_search_query(question, usage)
    context = tools.search_news(search_query)
    prompt = (
        f"다음은 관련 뉴스 기사 목록이다:\n{context}\n\n"
        f"{PREMISE_CHECK_RULE}\n{ANSWER_RULE}\n\n"
        f"이 기사들만 근거로 질문에 답하고, 사용한 기사 번호를 [1] 같은 형식으로 인용해줘. "
        f"기사에 없는 내용(예: 정확한 재무 수치)은 모른다고 답해. 질문: {question}"
    )
    answer = _ask(prompt, usage)
    elapsed = time.perf_counter() - start
    _log_experiment("news", question, f"[검색어: {search_query}]\n\n{prompt}", answer, elapsed)
    return {"answer": answer, "context": f"[search_news 검색어: {search_query}]\n{context}", "usage": usage}


def answer_with_all_tools(question: str, company: str) -> dict:
    """모든 도구(재무·뉴스·주가·업종 비교)를 질문 내용과 무관하게 항상 한 번씩 불러 답한다. {"answer", "context"} 반환.

    도구 인자는 고정: 재무는 최신 보고서, 주가는 최근 30일, 업종 비교는 대상 회사의 영업이익률.

    "도구 선택 능력이 있는 게 그냥 도구를 다 부르는 것보다 나은가?"를 보여주기 위한
    비교군 — 에이전트가 이기면 단순히 "정보가 많아서"가 아니라 "필요한 걸 골라 쓰고
    필요하면 재시도하는 능력" 덕분이라는 걸 뒷받침한다.
    """
    usage: dict = {}
    start = time.perf_counter()
    end = date.today().isoformat()
    news_query = _build_search_query(question, usage)
    dart_context = tools.get_financial_data(company)
    news_context = tools.search_news(news_query)
    try:
        price_context = tools.get_stock_price(company, (date.today() - timedelta(days=30)).isoformat(), end)
    except Exception as e:  # noqa: BLE001 — 비상장사 등으로 주가 조회가 아예 불가능한 경우
        price_context = f"조회 불가: {e}"
    peer_context = tools.compare_peers(company, "영업이익률")

    context = (
        f"[DART 재무 데이터]\n{dart_context}\n\n"
        f"[관련 뉴스 (검색어: {news_query})]\n{news_context}\n\n"
        f"[주가 데이터 (최근 30일)]\n{price_context}\n\n"
        f"[같은 업종 비교 (영업이익률)]\n{peer_context}"
    )
    prompt = (
        f"다음은 여러 출처의 데이터다:\n{context}\n\n"
        f"{PREMISE_CHECK_RULE}\n{ANSWER_RULE}\n\n"
        f"이 데이터만 근거로 질문에 답해줘. 관련 없는 출처는 무시하고, 데이터에 없는 "
        f"내용은 모른다고 답해. 질문: {question}"
    )
    answer = _ask(prompt, usage)
    elapsed = time.perf_counter() - start
    _log_experiment("all_tools", question, prompt, answer, elapsed)
    return {"answer": answer, "context": context, "usage": usage}
