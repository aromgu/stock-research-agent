"""Phase 4: DART/News/Price Tool 정의 (OpenAI function calling용 스키마 + 실행 로직).

Planner가 함수 호출로 직접 부를 수 있는 형태(문자열 인자 -> 문자열 결과)로 감싼다.
Phase 3 baseline도 이 함수들을 그대로 쓴다 — ablation에서 도구 구현은 같고
"도구를 어떻게 조합하느냐"만 달라야 공정한 비교가 되기 때문.
"""

import sqlite3
from datetime import date, timedelta
from pathlib import Path

from ..data import dart_client as dc
from ..data import news_client as nc
from ..data import price_client as pc

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "financials.db"

# reprt_code는 문자열/숫자로 정렬해도 분기 순서가 안 나옴 (1분기=11013 > 반기=11012).
_PERIOD_RANK = {"11013": 1, "11012": 2, "11014": 3, "11011": 4}
_PERIOD_LABEL = {"11013": "1분기보고서", "11012": "반기보고서", "11014": "3분기보고서", "11011": "사업보고서(연간)"}
_PERIOD_ARG_TO_CODE = {"annual": "11011", "q1": "11013", "h1": "11012", "q3": "11014"}
# 보고서 기간의 종료월 (전부 1월부터 누적 - 12월 결산 기업 기준, 삼성전자/SK하이닉스 해당)
_PERIOD_END_MONTH = {"11013": 3, "11012": 6, "11014": 9, "11011": 12}

_DEFAULT_PRICE_WINDOW_DAYS = 30
_MAX_DAILY_CLOSES_SHOWN = 40  # 기간이 길면 일별 종가 나열이 너무 길어져서 생략


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
            SELECT account_name, sj_name, thstrm_amount, thstrm_add_amount, frmtrm_amount, frmtrm_add_amount
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
                "frmtrm_amount": r[4],
                "frmtrm_add_amount": r[5],
                "bsns_year": year,
                "reprt_code": reprt_code,
            }
            for r in rows
        ]
    finally:
        conn.close()


def _won(v: int | None) -> str:
    return f"{v:,}원" if v is not None else "없음"


def _describe_row(f: dict) -> str:
    """전제 검증("왜 줄었어?" 등)에 쓰도록 비교 기간 값도 함께 적는다.

    DART 규칙상 비교 기간(frmtrm)의 의미가 표 종류마다 다르다:
    재무상태표(BS)는 시점 스냅샷이라 "전기말", 손익계산서(PL)는 "전년 동기".
    """
    if f["sj_name"] == "재무상태표":
        return f"- {f['account_name']}: 기말 시점 기준 {_won(f['thstrm_amount'])} (전기말 {_won(f['frmtrm_amount'])})"
    if f["reprt_code"] == "11011":
        return f"- {f['account_name']}: 연간 누적 {_won(f['thstrm_amount'])} (전년 연간 {_won(f['frmtrm_amount'])})"
    line = f"- {f['account_name']}: 해당 분기 단독 {_won(f['thstrm_amount'])}"
    prior = f"전년 동기 단독 {_won(f['frmtrm_amount'])}"
    if f["thstrm_add_amount"] is not None:
        line += f" / 연초~해당 분기 누적 {_won(f['thstrm_add_amount'])}"
        prior += f" / 전년 동기 누적 {_won(f['frmtrm_add_amount'])}"
    return f"{line} (비교: {prior})"


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
    end_month = _PERIOD_END_MONTH[reprt_code]
    period_range = f"{year}-01-01 ~ {year}-{end_month:02d}-{_month_end_day(end_month)}"
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


def get_stock_price(company: str, start: str | None = None, end: str | None = None) -> str:
    """Price Tool: 일별 주가 시세 + 기간 수익률 + PER/PBR/시가총액 (pykrx, KRX 데이터).

    start/end(YYYY-MM-DD) 미지정 시 end=오늘, start=end-30일.
    기간 수익률 = 기간 첫 거래일 종가 대비 마지막 거래일 종가 (채점기도 같은 정의를 씀).
    """
    end = end or date.today().isoformat()
    start = start or (date.fromisoformat(end) - timedelta(days=_DEFAULT_PRICE_WINDOW_DAYS)).isoformat()
    ticker = dc.get_stock_code(company)
    rows = pc.get_ohlcv(ticker, start, end)
    if not rows:
        return f"{company}({ticker}): {start} ~ {end} 기간의 시세 데이터가 없습니다 (휴장 기간이거나 날짜 범위 오류)."

    first, last = rows[0], rows[-1]
    ret = (last["close"] / first["close"] - 1) * 100
    hi = max(rows, key=lambda r: r["high"])
    lo = min(rows, key=lambda r: r["low"])
    lines = [
        f"{company}({ticker}) 주가 {first['date']} ~ {last['date']} (거래일 {len(rows)}일)",
        f"- 첫 거래일 종가({first['date']}): {first['close']:,}원",
        f"- 마지막 거래일 종가({last['date']}): {last['close']:,}원",
        f"- 기간 수익률(첫 거래일 종가 대비 마지막 거래일 종가): {ret:+.2f}%",
        f"- 기간 최고가 {hi['high']:,}원({hi['date']}) / 최저가 {lo['low']:,}원({lo['date']})",
    ]
    if len(rows) <= _MAX_DAILY_CLOSES_SHOWN:
        lines.append("- 일별 종가: " + ", ".join(f"{r['date'][5:]} {r['close']:,}" for r in rows))

    fund = pc.get_fundamental(ticker, last["date"])
    if fund:
        lines.append(
            f"- 밸류에이션({fund['date']} 기준, KRX 제공값): PER {fund['PER']:.2f}배, PBR {fund['PBR']:.2f}배, "
            f"EPS {fund['EPS']:,.0f}원, BPS {fund['BPS']:,.0f}원, 배당수익률 {fund['DIV']:.2f}%"
        )
    else:
        lines.append("- 밸류에이션(PER/PBR): 조회 불가 (KRX 로그인 정보 없음)")
    cap = pc.get_market_cap(ticker, last["date"])
    if cap:
        lines.append(f"- 시가총액({cap['date']} 기준): {cap['market_cap']:,}원 (상장주식수 {cap['shares']:,}주)")
    return "\n".join(lines)


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_financial_data",
            "description": (
                "기업의 DART 공시 재무 데이터(매출액/영업이익/자산 등)를 조회한다. 전년 동기(손익) 또는 "
                "전기말(재무상태) 비교값도 함께 준다. 현재 삼성전자, SK하이닉스만 지원."
            ),
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
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": (
                "상장 종목의 일별 주가, 기간 수익률, 기간 최고/최저가, PER/PBR/시가총액을 조회한다. "
                "주가가 실제로 올랐는지/내렸는지 확인하거나, 밸류에이션(비싼지/싼지)을 판단할 때 사용."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {"type": "string", "description": "회사명 또는 종목코드 (예: 삼성전자, 005930)"},
                    "start": {"type": "string", "description": "시작일 YYYY-MM-DD (선택, 기본값: 종료일 30일 전)"},
                    "end": {"type": "string", "description": "종료일 YYYY-MM-DD (선택, 기본값: 오늘)"},
                },
                "required": ["company"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "get_financial_data": get_financial_data,
    "search_news": search_news,
    "get_stock_price": get_stock_price,
}
