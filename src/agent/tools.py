"""Phase 4: DART/News Tool 정의 (OpenAI function calling용 스키마 + 실행 로직).

Phase 3 baseline.py와 동일한 DART 조회/라벨링 로직을 쓰되, Planner가 함수 호출로
직접 부를 수 있는 형태(문자열 인자 -> 문자열 결과)로 감싼다.
"""

import sqlite3
from pathlib import Path

from ..data import dart_client as dc
from ..data import news_client as nc

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "financials.db"

# reprt_code는 문자열/숫자로 정렬해도 분기 순서가 안 나옴 (1분기=11013 > 반기=11012) — baseline.py와 동일한 함정.
_PERIOD_RANK = {"11013": 1, "11012": 2, "11014": 3, "11011": 4}
_PERIOD_LABEL = {"11013": "1분기보고서", "11012": "반기보고서", "11014": "3분기보고서", "11011": "사업보고서(연간)"}
_PERIOD_ARG_TO_CODE = {"annual": "11011", "q1": "11013", "h1": "11012", "q3": "11014"}
# 보고서 기간의 종료월 (전부 1월부터 누적 - 12월 결산 기업 기준, 삼성전자/SK하이닉스 해당)
_PERIOD_END_MONTH = {"11013": 3, "11012": 6, "11014": 9, "11011": 12}


def _fetch_financials(corp_code: str, reprt_code: str | None, n_accounts: int = 20) -> list[dict]:
    """로컬 DB(Phase 1 수집분)에서 재무 항목을 가져온다. reprt_code=None이면 가장 최근 보고서."""
    conn = sqlite3.connect(DB_PATH)
    try:
        periods = conn.execute(
            "SELECT DISTINCT bsns_year, reprt_code FROM financials WHERE corp_code=?", (corp_code,)
        ).fetchall()
        if reprt_code is not None:
            periods = [p for p in periods if p[1] == reprt_code]
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


def _describe_row(f: dict) -> str:
    if f["sj_name"] == "재무상태표":  # BS: 시점 스냅샷
        return f"- {f['account_name']}: 기말 시점 기준 {f['thstrm_amount']:,}원"
    label = "연간 누적" if f["reprt_code"] == "11011" else "해당 분기 단독"
    line = f"- {f['account_name']}: {label} {f['thstrm_amount']:,}원"
    if f["thstrm_add_amount"] is not None:
        line += f" / 연초~해당 분기 누적 {f['thstrm_add_amount']:,}원"
    return line


def get_financial_data(company: str, period: str = "latest") -> str:
    """DART Tool: 로컬 DB(Phase 1)에 저장된 기업 재무 데이터를 조회한다.

    period: "latest"(기본, 가장 최근 보고서) | "annual" | "q1" | "h1" | "q3"
    현재 삼성전자/SK하이닉스 최근 4개 보고서만 로컬 DB에 있음.
    """
    reprt_code = _PERIOD_ARG_TO_CODE.get(period.lower())  # "latest" 등 매칭 안 되면 None -> 최근 보고서

    corp_code = dc.get_corp_code(company)
    overview = dc.get_company_overview(corp_code)
    facts = _fetch_financials(corp_code, reprt_code)

    if not facts:
        return f"{overview['corp_name']}: 요청한 기간의 재무 데이터가 로컬 DB에 없음 (현재 삼성전자/SK하이닉스 최근 4개 보고서만 지원)"

    year, reprt_code = facts[0]["bsns_year"], facts[0]["reprt_code"]
    period_label = f"{year}년 {_PERIOD_LABEL[reprt_code]}"
    period_range = f"{year}-01-01 ~ {year}-{_PERIOD_END_MONTH[reprt_code]:02d}-{_month_end_day(_PERIOD_END_MONTH[reprt_code])}"
    header = f"{overview['corp_name']} ({period_label}, 기간 {period_range}, 연결재무제표 기준)"
    return header + "\n" + "\n".join(_describe_row(f) for f in facts)


def _month_end_day(month: int) -> int:
    return {3: 31, 6: 30, 9: 30, 12: 31}[month]


def search_news(query: str, after: str | None = None, before: str | None = None) -> str:
    """News Tool: 뉴스를 실시간 검색하고 쿼리 시점에 bge-m3로 재정렬한 상위 5개를 반환한다.

    after/before(YYYY-MM-DD)를 주면 그 날짜 범위의 기사만, 안 주면 최근 30일 기본값.
    """
    # candidate_display=20이면 인기 검색어는 최근 며칠치로 다 채워져 과거 기간을 못 봄(직접 확인).
    # 100(NAVER 요청당 최대치)까지 받아야 date range 필터가 실제로 쓸모 있음.
    articles = nc.search_news_reranked(query, candidate_display=100, top_k=5, after=after, before=before)
    if not articles:
        if after or before:
            return (
                f"{after or '(제한없음)'} ~ {before or '(제한없음)'} 기간의 기사를 찾지 못했습니다. "
                "NAVER 검색이 인기 검색어는 최근 며칠치로 결과가 꽉 차서 몇 달 전 과거까지 도달하지 "
                "못하는 경우가 흔합니다 — 날짜 범위를 좁히거나(더 최근으로), 검색어를 바꿔서 다시 시도해보세요."
            )
        return "관련 뉴스를 찾지 못했습니다."
    return "\n".join(f"- [{i + 1}] ({a['pub_date']}) {a['title']}: {a['description']}" for i, a in enumerate(articles))


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_financial_data",
            "description": "기업의 DART 공시 재무 데이터(매출액/영업이익/자산 등)를 조회한다. 현재 삼성전자, SK하이닉스만 지원.",
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {"type": "string", "description": "회사명 또는 종목코드 (예: 삼성전자, 005930)"},
                    "period": {
                        "type": "string",
                        "enum": ["latest", "annual", "q1", "h1", "q3"],
                        "description": "조회할 보고서 기간. latest=가장 최근 보고서(기본값), annual=연간, q1=1분기, h1=반기, q3=3분기",
                    },
                },
                "required": ["company"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": (
                "관련 뉴스를 검색한다. 실적 원인, 업계 동향 등 정성적 맥락이 필요할 때 사용. "
                "특정 과거 기간(예: 실적 발표 기간)에 대한 원인을 찾을 때는 '오늘 기준 최근 뉴스'가 "
                "아니라 반드시 after/before로 그 기간을 지정해야 한다 — 원인이 된 사건이 오늘이 아니라 "
                "그 분기 중간에 있었을 수 있다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "검색어"},
                    "after": {"type": "string", "description": "이 날짜(YYYY-MM-DD) 이후 기사만 (선택)"},
                    "before": {"type": "string", "description": "이 날짜(YYYY-MM-DD) 이전 기사만 (선택)"},
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "get_financial_data": get_financial_data,
    "search_news": search_news,
}
