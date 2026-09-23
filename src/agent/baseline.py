"""Phase 3: 단일 도구 RAG 베이스라인 (DART만 / News만).

Phase 4의 멀티홉 에이전트(플래너+동적 도구선택) 전 단계로, 도구를 하나만 쓰는
가장 단순한 RAG를 먼저 구축해 baseline 성능을 잰다 (Notion 계획서 §1.4 ablation 비교용).
"""

import sqlite3
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from ..data import dart_client as dc
from ..data import news_client as nc

MODEL = "gpt-5.4-nano"  # 저렴한 모델로 고정 (베이스라인 단계에서 비용 최소화)
DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "financials.db"
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


# reprt_code는 문자열/숫자로 정렬해도 분기 순서가 안 나옴 (1분기=11013 > 반기=11012).
# 연도 내 실제 시간 순서로 정렬하기 위한 순위 매핑.
_PERIOD_RANK = {"11013": 1, "11012": 2, "11014": 3, "11011": 4}
_PERIOD_LABEL = {"11013": "1분기보고서", "11012": "반기보고서", "11014": "3분기보고서", "11011": "사업보고서(연간)"}

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _fetch_latest_financials(corp_code: str, n_accounts: int = 20) -> list[dict]:
    """로컬 DB(Phase 1 수집분)에서 가장 최근 보고서의 연결(CFS) 재무 항목을 가져온다."""
    conn = sqlite3.connect(DB_PATH)
    try:
        periods = conn.execute(
            "SELECT DISTINCT bsns_year, reprt_code FROM financials WHERE corp_code=?",
            (corp_code,),
        ).fetchall()
        if not periods:
            return []
        year, reprt_code = max(periods, key=lambda p: (int(p[0]), _PERIOD_RANK[p[1]]))
        rows = conn.execute(
            """
            SELECT account_name, sj_name, thstrm_amount, thstrm_add_amount
            FROM financials
            WHERE corp_code=? AND bsns_year=? AND reprt_code=? AND fs_div='CFS'
            LIMIT ?
            """,
            (corp_code, year, reprt_code, n_accounts),
        ).fetchall()
        return [
            {
                "account_name": r[0],
                "sj_name": r[1],
                "thstrm_amount": r[2],
                "thstrm_add_amount": r[3],
                "bsns_year": year,
                "reprt_code": reprt_code,
            }
            for r in rows
        ]
    finally:
        conn.close()


def answer_with_dart(question: str, name_or_stock_code: str) -> str:
    """DART 데이터만 근거로 질문에 답한다 (뉴스 없이). 로컬 DB에 없으면 빈 컨텍스트로 답함."""
    start = time.perf_counter()  # 검색(DART 조회)부터 LLM 호출까지 전체 소요시간
    corp_code = dc.get_corp_code(name_or_stock_code)
    overview = dc.get_company_overview(corp_code)
    facts = _fetch_latest_financials(corp_code)

    if facts:
        period_label = f"{facts[0]['bsns_year']}년 {_PERIOD_LABEL[facts[0]['reprt_code']]}"
        header = f"{overview['corp_name']} ({period_label}, 연결재무제표 기준)"
    else:
        header = f"{overview['corp_name']} (로컬 DB에 저장된 재무 데이터 없음)"

    def _describe(f: dict) -> str:
        if f["sj_name"] == "재무상태표":  # BS: 시점 스냅샷 — "분기 단독/누적" 개념 자체가 안 맞음
            return f"- {f['account_name']}: 기말 시점 기준 {f['thstrm_amount']:,}원"
        # PL(손익계산서): 연간보고서는 thstrm_amount 자체가 이미 연간 누적치라 "분기 단독"이 틀림
        thstrm_label = "연간 누적" if f["reprt_code"] == "11011" else "해당 분기 단독"
        line = f"- {f['account_name']}: {thstrm_label} {f['thstrm_amount']:,}원"
        if f["thstrm_add_amount"] is not None:
            line += f" / 연초~해당 분기 누적 {f['thstrm_add_amount']:,}원"
        return line

    lines = [_describe(f) for f in facts]
    context = header + "\n" + "\n".join(lines)
    prompt = (
        f"다음은 DART 공시 재무 데이터다:\n{context}\n\n"
        f"이 데이터만 근거로 질문에 답해줘. 데이터에 없는 내용은 모른다고 답해. 질문: {question}"
    )
    resp = _get_client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=300,
    )
    elapsed = time.perf_counter() - start
    answer = resp.choices[0].message.content
    _log_experiment("dart", question, prompt, answer, elapsed)
    return answer


def answer_with_news(question: str, top_k: int = 5) -> str:
    """뉴스 검색 결과만 근거로 질문에 답한다 (DART 없이)."""
    start = time.perf_counter()  # 검색(뉴스 API+재정렬)부터 LLM 호출까지 전체 소요시간
    articles = nc.search_news_reranked(question, candidate_display=20, top_k=top_k)

    context = "\n".join(f"- [{i+1}] {a['title']}: {a['description']}" for i, a in enumerate(articles))
    prompt = (
        f"다음은 관련 뉴스 기사 목록이다:\n{context}\n\n"
        f"이 기사들만 근거로 질문에 답하고, 사용한 기사 번호를 [1] 같은 형식으로 인용해줘. 질문: {question}"
    )
    resp = _get_client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=300,
    )
    elapsed = time.perf_counter() - start
    answer = resp.choices[0].message.content
    _log_experiment("news", question, prompt, answer, elapsed)
    return answer
