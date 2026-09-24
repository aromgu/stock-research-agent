"""Phase 3: 단일 도구 RAG 베이스라인 (DART만 / News만 / 항상 3도구 전부).

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
from .prompts import PREMISE_CHECK_RULE

MODEL = "gpt-5.4-nano"  # 저렴한 모델로 고정 (베이스라인 단계에서 비용 최소화)
LOG_PATH = Path(__file__).resolve().parent.parent.parent / "logs" / "baseline_experiments.md"


def _log_experiment(tool: str, question: str, prompt: str, answer: str, elapsed_sec: float) -> None:
    """실험 입출력을 logs/baseline_experiments.md에 사람이 읽기 좋은 형태로 append."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = (
        f"## [{timestamp}] {tool.upper()} · {MODEL} · {elapsed_sec:.1f}초\n\n"
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


def _ask(prompt: str) -> str:
    resp = _get_client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=300,
    )
    return resp.choices[0].message.content


def answer_with_dart(question: str, name_or_stock_code: str) -> str:
    """DART 데이터만 근거로 질문에 답한다 (뉴스/주가 없이)."""
    start = time.perf_counter()
    context = tools.get_financial_data(name_or_stock_code)
    prompt = (
        f"다음은 DART 공시 재무 데이터다:\n{context}\n\n"
        f"{PREMISE_CHECK_RULE}\n\n"
        f"이 데이터만 근거로 질문에 답해줘. 데이터에 없는 내용(예: 주가, 뉴스 맥락)은 "
        f"모른다고 답해. 질문: {question}"
    )
    answer = _ask(prompt)
    elapsed = time.perf_counter() - start
    _log_experiment("dart", question, prompt, answer, elapsed)
    return answer


def _build_search_query(question: str) -> str:
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
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=50,
    )
    return resp.choices[0].message.content.strip()


def answer_with_news(question: str, top_k: int = 5) -> str:
    """뉴스 검색 결과만 근거로 질문에 답한다 (DART/주가 없이)."""
    start = time.perf_counter()  # 검색어 생성부터 LLM 최종 답변까지 전체 소요시간
    search_query = _build_search_query(question)
    context = tools.search_news(search_query)
    prompt = (
        f"다음은 관련 뉴스 기사 목록이다:\n{context}\n\n"
        f"{PREMISE_CHECK_RULE}\n\n"
        f"이 기사들만 근거로 질문에 답하고, 사용한 기사 번호를 [1] 같은 형식으로 인용해줘. "
        f"기사에 없는 내용(예: 정확한 재무 수치)은 모른다고 답해. 질문: {question}"
    )
    answer = _ask(prompt)
    elapsed = time.perf_counter() - start
    _log_experiment("news", question, f"[검색어: {search_query}]\n\n{prompt}", answer, elapsed)
    return answer


def answer_with_all_tools(question: str, company: str) -> str:
    """DART+뉴스+주가 세 도구를 질문 내용과 무관하게 항상 전부 불러 답한다.

    "도구 선택 능력이 있는 게 그냥 도구를 다 부르는 것보다 나은가?"를 보여주기 위한
    비교군 — 에이전트가 이기면 단순히 "정보가 많아서"가 아니라 "필요한 걸 골라 쓰고
    필요하면 재시도하는 능력" 덕분이라는 걸 뒷받침한다.
    """
    start = time.perf_counter()
    end = date.today().isoformat()
    news_query = _build_search_query(question)
    dart_context = tools.get_financial_data(company)
    news_context = tools.search_news(news_query)
    try:
        price_context = tools.get_stock_price(company, (date.today() - timedelta(days=30)).isoformat(), end)
    except Exception as e:  # noqa: BLE001 — 비상장사 등으로 주가 조회가 아예 불가능한 경우
        price_context = f"조회 불가: {e}"

    context = (
        f"[DART 재무 데이터]\n{dart_context}\n\n"
        f"[관련 뉴스 (검색어: {news_query})]\n{news_context}\n\n"
        f"[주가 데이터 (최근 30일)]\n{price_context}"
    )
    prompt = (
        f"다음은 세 가지 출처의 데이터다:\n{context}\n\n"
        f"{PREMISE_CHECK_RULE}\n\n"
        f"이 데이터만 근거로 질문에 답해줘. 관련 없는 출처는 무시하고, 데이터에 없는 "
        f"내용은 모른다고 답해. 질문: {question}"
    )
    answer = _ask(prompt)
    elapsed = time.perf_counter() - start
    _log_experiment("all_tools", question, prompt, answer, elapsed)
    return answer
