"""Phase 4: DART/News/Price Tool 정의 (OpenAI function calling용 스키마 + 실행 로직).

Planner가 함수 호출로 직접 부를 수 있는 형태(문자열 인자 -> 문자열 결과)로 감싼다.
Phase 3 baseline도 이 함수들을 그대로 쓴다 — ablation에서 도구 구현은 같고
"도구를 어떻게 조합하느냐"만 달라야 공정한 비교가 되기 때문.
"""

import re
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from ..data import collect_financials as cf
from ..data import dart_client as dc
from ..data import news_client as nc
from ..data import price_client as pc
from ..data import universe as uv

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "financials.db"

# reprt_code는 문자열/숫자로 정렬해도 분기 순서가 안 나옴 (1분기=11013 > 반기=11012).
_PERIOD_RANK = {"11013": 1, "11012": 2, "11014": 3, "11011": 4}
_PERIOD_LABEL = {"11013": "1분기보고서", "11012": "반기보고서", "11014": "3분기보고서", "11011": "사업보고서(연간)"}
_PERIOD_ARG_TO_CODE = {"annual": "11011", "q1": "11013", "h1": "11012", "q3": "11014"}
# 보고서 기간의 종료월 (1월부터 누적 - 12월 결산 기업 기준. 다른 결산월 회사는 get_financial_data에서 따로 처리)
_PERIOD_END_MONTH = {"11013": 3, "11012": 6, "11014": 9, "11011": 12}

_DEFAULT_PRICE_WINDOW_DAYS = 30
_MAX_DAILY_CLOSES_SHOWN = 40  # 기간이 길면 일별 종가 나열이 너무 길어져서 생략


def _available_periods(corp_code: str) -> list[tuple[str, str]]:
    """로컬 DB에 있는 (사업연도, 보고서코드) 목록, 시간순."""
    conn = sqlite3.connect(DB_PATH)
    try:
        periods = conn.execute(
            "SELECT DISTINCT bsns_year, reprt_code FROM financials WHERE corp_code=?", (corp_code,)
        ).fetchall()
    finally:
        conn.close()
    return sorted(periods, key=lambda p: (int(p[0]), _PERIOD_RANK[p[1]]))


def _describe_available(periods: list[tuple[str, str]]) -> str:
    return ", ".join(f"{y}년 {_PERIOD_LABEL[c]}" for y, c in periods)


def _fetch_financials(
    corp_code: str, reprt_code: str | None, n_accounts: int = 20, year: str | None = None
) -> list[dict]:
    """로컬 DB(Phase 1 수집분)에서 재무 항목을 가져온다. reprt_code/year가 None이면 조건 없이 가장 최근 보고서."""
    conn = sqlite3.connect(DB_PATH)
    try:
        periods = _available_periods(corp_code)
        if reprt_code is not None:
            periods = [p for p in periods if p[1] == reprt_code]
        if year is not None:
            periods = [p for p in periods if p[0] == str(year)]
        if not periods:
            return []
        year, reprt_code = max(periods, key=lambda p: (int(p[0]), _PERIOD_RANK[p[1]]))
        # 연결재무제표(CFS)를 우선 쓰고, 자회사가 없어 별도재무제표(OFS)만 공시하는 회사는 OFS로 대체
        for fs_div in ("CFS", "OFS"):
            rows = conn.execute(
                """
                SELECT account_name, sj_name, thstrm_amount, thstrm_add_amount, frmtrm_amount, frmtrm_add_amount
                FROM financials
                WHERE corp_code=? AND bsns_year=? AND reprt_code=? AND fs_div=?
                LIMIT ?
                """,
                (corp_code, year, reprt_code, fs_div, n_accounts),
            ).fetchall()
            if rows:
                break
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
                "fs_div": fs_div,
            }
            for r in rows
        ]
    finally:
        conn.close()


def _ensure_reports(corp_code: str, reprt_code: str | None, year: str | None) -> None:
    """로컬 DB에 없는 회사·기간이면 DART에서 받아 DB에 저장한다 (한 번 받으면 이후엔 DB에서 바로 읽음).

    처음 보는 회사는 최근 보고서 4개를 받고, 특정 연도·기간을 요청했는데 없으면 그것만 받아본다.
    DART는 무료지만 호출마다 1~2초 걸려서, 필요한 것만 받는다.
    """
    periods = _available_periods(corp_code)
    if not periods:
        cf.collect_recent_quarters(corp_code)
        periods = _available_periods(corp_code)
    matches = [p for p in periods if (reprt_code is None or p[1] == reprt_code) and (year is None or p[0] == str(year))]
    if matches or (reprt_code is None and year is None):
        return
    years = [str(year)] if year else [str(date.today().year), str(date.today().year - 1)]
    codes = [reprt_code] if reprt_code else sorted(_PERIOD_RANK, key=_PERIOD_RANK.get, reverse=True)
    for y in years:
        for c in codes:
            if (y, c) not in periods and cf.fetch_report(corp_code, y, c):
                return


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


def get_financial_data(company: str, period: str = "latest", year: str | int | None = None) -> str:
    """DART Tool: 로컬 DB(Phase 1)에 저장된 기업 재무 데이터를 조회한다.

    period: "latest"(기본, 가장 최근 보고서) | "annual" | "q1" | "h1" | "q3"
    year: 사업연도(예: 2025). 생략하면 해당 period의 가장 최근 연도.
    결과 첫머리에 조회 가능한 보고서 목록을 붙인다 — 예전엔 연도 파라미터도, 목록도 없어서
    "2025년 1~9월"을 물었을 때 에이전트가 latest(2026 반기)만 보고 "자료 없음"으로 포기했다.
    """
    reprt_code = _PERIOD_ARG_TO_CODE.get(period.lower())  # "latest" 등 매칭 안 되면 None -> 최근 보고서
    year = str(year) if year else None

    try:
        corp_code = dc.get_corp_code(company)
    except KeyError:
        candidates = ", ".join(dc.suggest_listed_names(company)) or "없음"
        return f"'{company}'에 해당하는 회사를 DART에서 찾지 못함. 비슷한 상장사 이름: {candidates} (정확한 이름으로 다시 조회)"
    overview = dc.get_company_overview(corp_code)
    _ensure_reports(corp_code, reprt_code, year)
    available = f"(조회 가능한 보고서: {_describe_available(_available_periods(corp_code)) or '없음'})"
    facts = _fetch_financials(corp_code, reprt_code, year=year)

    if not facts:
        if reprt_code is None and year is None:
            return (
                f"{overview['corp_name']}: DART 재무제표 API에서 최근 보고서를 찾지 못함 "
                "(상장폐지, 재무제표 미제출, 또는 이 API가 지원하지 않는 공시 형식일 수 있음)"
            )
        requested = f"{year}년 " if year else ""
        return f"{overview['corp_name']}: 요청한 {requested}{period} 보고서가 DART에 없음 (아직 공시 전일 수 있음) {available}"

    year, reprt_code = facts[0]["bsns_year"], facts[0]["reprt_code"]
    period_label = f"{year}년 {_PERIOD_LABEL[reprt_code]}"
    fs_label = "연결재무제표" if facts[0]["fs_div"] == "CFS" else "별도재무제표(연결 없음)"
    if overview.get("acc_mt") == "12":
        end_month = _PERIOD_END_MONTH[reprt_code]
        period_note = f"기간 {year}-01-01 ~ {year}-{end_month:02d}-{_month_end_day(end_month)}"
    else:
        # 12월 결산이 아니면 회계연도가 1월에 시작하지 않아 날짜 범위를 계산하지 않는다
        period_note = f"{overview.get('acc_mt')}월 결산 회사라 분기는 회계연도 기준"
    header = f"{overview['corp_name']} ({period_label}, {period_note}, {fs_label} 기준) {available}"
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
    try:
        ticker = dc.get_stock_code(company)
    except KeyError as e:
        candidates = ", ".join(dc.suggest_listed_names(company)) or "없음"
        return f"{e.args[0]} 비슷한 상장사 이름: {candidates} (정확한 이름으로 다시 조회)"
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


# 업종 비교 지표: (분자 계정, 분모 계정, 종류). 손익 항목은 "연초~해당 분기 누적" 기준 (1분기·연간은 당기 값이 곧 누적)
_PEER_METRICS = {
    "영업이익률": ("영업이익", "매출액", "ratio"),
    "순이익률": ("당기순이익(손실)", "매출액", "ratio"),
    "부채비율": ("부채총계", "자본총계", "ratio"),
    "매출액 증가율": ("매출액", None, "growth"),
    "영업이익 증가율": ("영업이익", None, "growth"),
    "매출액": ("매출액", None, "amount"),
    "영업이익": ("영업이익", None, "amount"),
}


def _cumulative(row: dict, prior: bool = False) -> int | None:
    if prior:
        return row["frmtrm_add_amount"] if row["frmtrm_add_amount"] is not None else row["frmtrm_amount"]
    return row["thstrm_add_amount"] if row["thstrm_add_amount"] is not None else row["thstrm_amount"]


def _peer_metric(corp_code: str, metric: str) -> tuple[float | None, str]:
    """(지표 값, 보고서 라벨). 계정이 없으면(금융사의 매출액 등) 값은 None."""
    num_name, den_name, kind = _PEER_METRICS[metric]
    facts = _fetch_financials(corp_code, None)
    if not facts:
        return None, "재무 데이터 없음"
    rows = {f["account_name"]: f for f in facts}
    label = f"{facts[0]['bsns_year']}년 {_PERIOD_LABEL[facts[0]['reprt_code']]}"
    num = rows.get(num_name)
    if num is None:
        return None, label
    if kind == "amount":
        return _cumulative(num), label
    if kind == "growth":
        cur, prev = _cumulative(num), _cumulative(num, prior=True)
        return ((cur / prev - 1) * 100 if cur is not None and prev else None), label
    den = rows.get(den_name)
    a, b = _cumulative(num), (_cumulative(den) if den else None)
    return ((a / b * 100) if a is not None and b else None), label


def _has_revenue(corp_code: str) -> bool:
    return any(f["account_name"] == "매출액" for f in _fetch_financials(corp_code, None))


_GENERIC_WORDS = {"반도체", "업체", "업종", "회사", "기업", "관련", "종목", "중"}


def _match_segments(target: str) -> list[str]:
    """업종 표현을 반도체 세부 업종에 맞춘다: "반도체 소재" → 소재, "장비주" → 전공정·후공정 장비.

    업종 이름을 통째로 말했으면(예: "전공정 장비") 그 업종만 쓴다 — 단어 단위로만 보면 "장비"가 후공정까지 잡힌다.
    """
    exact = [s for s in uv.SEMICONDUCTOR_SEGMENTS if s in target]
    if exact:
        return exact
    words = {w[:-1] if w.endswith("주") and len(w) > 2 else w for w in re.findall(r"[가-힣A-Za-z]+", target)}
    words -= _GENERIC_WORDS
    return [s for s in uv.SEMICONDUCTOR_SEGMENTS if words & set(re.findall(r"[가-힣A-Za-z]+", s))]


def _resolve_peer_group(target: str) -> tuple[str, list[dict], str | None]:
    """(그룹 설명, 그룹 회사 목록, 기준 회사 corp_code). 업종명이면 그 업종, 회사명이면 그 회사가 속한 업종."""
    universe = uv.load_universe()
    # "반도체 소재", "장비주"처럼 업종 이름이 들어간 표현도 받는다 (예전엔 "소재"와 정확히 같아야만 인식해서
    # 에이전트가 "반도체 소재"로 부르면 업종을 못 찾았다). 여러 업종에 걸치면(예: "장비") 합쳐서 비교.
    segments = _match_segments(target)
    if segments:
        members = [c for c in universe if c.get("segment") in segments]
        return f"{uv.VALUE_CHAIN_LABEL} · {', '.join(segments)}", members, None
    for label in (uv.VALUE_CHAIN_LABEL, uv.KOSPI_TOP_LABEL):
        if target in label or label in target:
            return label, [c for c in universe if label in c["groups"]], None

    corp_code = dc.get_corp_code(target)  # 못 찾으면 KeyError → 호출부에서 후보 이름 안내
    me = next((c for c in universe if c["corp_code"] == corp_code), None)
    if me and me.get("segment"):
        return f"{uv.VALUE_CHAIN_LABEL} · {me['segment']}", [c for c in universe if c.get("segment") == me["segment"]], corp_code
    overview = dc.get_company_overview(corp_code)
    induty = me["induty_code"] if me else overview.get("induty_code", "")
    if me is None:  # 종목군 밖 회사도 비교에 포함 (필요하면 DART에서 받아옴)
        _ensure_reports(corp_code, None, None)
    same = [c for c in universe if c["induty_code"] == induty]
    if len(same) < 3:  # 같은 세부 업종 코드가 너무 적으면 한 단계 넓은 업종(앞 2자리)으로
        same = [c for c in universe if c["induty_code"][:2] == induty[:2]]
    # 금융지주(KB금융)와 일반 지주회사(SK)는 DART 업종 코드가 같다(64992). 금융사는 매출액 계정이 없으므로
    # 매출액 계정 유무가 같은 회사끼리만 묶어, 은행과 지주회사가 한 그룹에 섞이지 않게 한다.
    target_has_revenue = _has_revenue(corp_code)
    same = [c for c in same if _has_revenue(c["corp_code"]) == target_has_revenue]
    if me is None:
        same = [{"name": overview["corp_name"], "corp_code": corp_code}] + same
    return f"DART 업종코드 {induty[:2]}xx 계열 (핵심 종목군 기준)", same, corp_code


def compare_peers(target: str, metric: str = "영업이익률") -> str:
    """Peer Tool: 같은 업종 회사들을 재무 지표로 순위를 매긴다 (미리 수집한 핵심 종목군 기준).

    "장비주 중 영업이익률 1위", "SK하이닉스는 같은 업종 대비 부채비율이 높은 편이야?" 같은 질문용.
    target: 회사명(그 회사의 업종과 비교) 또는 업종명(메모리, 전공정 장비, 반도체 밸류체인, 코스피 시가총액 상위 등).
    """
    if metric not in _PEER_METRICS:
        return f"지원하지 않는 지표: {metric} (가능: {', '.join(_PEER_METRICS)})"
    try:
        group, members, target_code = _resolve_peer_group(target)
    except KeyError:
        candidates = ", ".join(dc.suggest_listed_names(target)) or "없음"
        groups = ", ".join([*uv.SEMICONDUCTOR_SEGMENTS, uv.VALUE_CHAIN_LABEL, uv.KOSPI_TOP_LABEL])
        return (
            f"'{target}'을(를) 업종명이나 회사명으로 찾지 못함. 사용 가능한 업종·그룹: {groups}. "
            f"비슷한 상장사 이름: {candidates}"
        )

    results = []
    for c in members:
        value, label = _peer_metric(c["corp_code"], metric)
        results.append((c, value, label))
    ranked = sorted((r for r in results if r[1] is not None), key=lambda r: r[1], reverse=True)
    missing = [r for r in results if r[1] is None]

    unit = "원" if _PEER_METRICS[metric][2] == "amount" else "%"
    lines = [
        f"비교 그룹: {group} ({len(members)}개사, 종목군 스냅샷 {uv.snapshot_date()} 기준)",
        f"지표: {metric} (각 회사의 가장 최근 보고서, 손익은 연초~해당 분기 누적 기준, 높은 순)",
    ]
    for i, (c, value, label) in enumerate(ranked, 1):
        mark = " ★질문 대상" if c["corp_code"] == target_code else ""
        shown = f"{value:,.0f}{unit}" if unit == "원" else f"{value:.2f}{unit}"
        lines.append(f"{i}. {c['name']}: {shown} ({label}){mark}")
    for c, _, label in missing:
        lines.append(f"- {c['name']}: 계산 불가 ({label}; 금융사처럼 해당 계정이 없는 회사)")
    # 이익률이 100%를 넘으면 매출보다 이익이 큰 것 — 지분법 이익이 큰 지주회사 등. 부채비율은 은행이면
    # 1,000%대가 정상이라 이 주의 문구를 붙이지 않는다.
    if metric in ("영업이익률", "순이익률") and any(abs(r[1]) > 100 for r in ranked):
        lines.append("※ 이익률이 100%를 넘는 회사는 지주회사처럼 매출 구조가 달라 단순 비교에 주의")
    periods = {r[2] for r in ranked}
    if len(periods) > 1:
        lines.append(f"※ 회사마다 최근 보고서 기간이 다름({', '.join(sorted(periods))}) — 직접 비교에 주의")
    return "\n".join(lines)


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_financial_data",
            "description": (
                "기업의 DART 공시 재무 데이터(매출액/영업이익/자산 등)를 조회한다. 전년 동기(손익) 또는 "
                "전기말(재무상태) 비교값도 함께 준다. DART에 공시하는 모든 회사를 지원 (처음 조회하는 회사는 "
                "DART에서 받아오느라 몇 초 걸림). 회사를 못 찾으면 비슷한 이름 후보를 알려주니 그 이름으로 다시 조회."
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
                    "year": {
                        "type": "string",
                        "description": "사업연도 (예: 2025). 질문에 특정 연도가 있으면 지정. 생략하면 해당 기간의 가장 최근 연도",
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

TOOL_SCHEMAS.append(
    {
        "type": "function",
        "function": {
            "name": "compare_peers",
            "description": (
                "같은 업종 회사들을 재무 지표로 순위를 매긴다. '업종 1위', '경쟁사 대비', '같은 업종에서 높은 편인지' 같은 "
                "비교 질문에 사용. 회사명을 주면 그 회사가 속한 업종과, 업종명을 주면 그 업종 전체와 비교한다. "
                "반도체 세부 업종: " + ", ".join(uv.SEMICONDUCTOR_SEGMENTS) + f". 그룹: {uv.VALUE_CHAIN_LABEL}, {uv.KOSPI_TOP_LABEL}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "회사명 또는 업종명 (예: 한미반도체, 전공정 장비)"},
                    "metric": {"type": "string", "enum": list(_PEER_METRICS), "description": "비교할 지표"},
                },
                "required": ["target", "metric"],
            },
        },
    }
)

TOOL_FUNCTIONS = {
    "get_financial_data": get_financial_data,
    "search_news": search_news,
    "get_stock_price": get_stock_price,
    "compare_peers": compare_peers,
}
