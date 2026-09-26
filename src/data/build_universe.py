"""핵심 종목군 스냅샷을 만들고, 각 회사의 최근 보고서 4개를 DART에서 미리 수집한다.

실행: python -m src.data.build_universe [--as-of 2026-09-23] [--top 30]
DART·KRX 모두 무료. 약 50개사 × 보고서 4개라 처음엔 몇 분 걸리고, 이미 수집한 회사는 건너뛴다.
"""

import argparse
import json
import sqlite3

from pykrx import stock

from . import collect_financials as cf
from . import dart_client as dc
from . import price_client  # noqa: F401 — import 시 .env 로드 + KRX 로그인 (시가총액 조회에 필요)
from .universe import KOSPI_TOP_LABEL, SEMICONDUCTOR_SEGMENTS, SNAPSHOT_PATH, VALUE_CHAIN_LABEL


def _kospi_top(as_of: str, top_n: int) -> list[str]:
    """as_of 기준 코스피 시가총액 상위 top_n 보통주의 종목코드 (우선주 제외)."""
    df = stock.get_market_cap(as_of.replace("-", ""), market="KOSPI").sort_values("시가총액", ascending=False)
    tickers = []
    for ticker in df.index:
        name = stock.get_market_ticker_name(ticker)
        if ticker[-1] != "0" or name.endswith(("우", "우B", "우C")):  # 우선주는 종목코드 끝자리가 0이 아님
            continue
        tickers.append(ticker)
        if len(tickers) == top_n:
            break
    return tickers


def build_snapshot(as_of: str, top_n: int) -> dict:
    companies: dict[str, dict] = {}

    def add(name_or_code: str, group: str, segment: str | None = None) -> None:
        entry = dc._find_entry(name_or_code)
        item = companies.setdefault(
            entry["corp_code"],
            {"name": entry["corp_name"], "stock_code": entry["stock_code"], "corp_code": entry["corp_code"], "groups": []},
        )
        if group not in item["groups"]:
            item["groups"].append(group)
        if segment:
            item["segment"] = segment

    for segment, names in SEMICONDUCTOR_SEGMENTS.items():
        for name in names:
            add(name, VALUE_CHAIN_LABEL, segment)
    for ticker in _kospi_top(as_of, top_n):
        add(ticker, KOSPI_TOP_LABEL)

    for item in companies.values():  # 시가총액 상위 종목끼리 업종 비교할 때 쓰는 DART 업종 코드(KSIC)
        item["induty_code"] = dc.get_company_overview(item["corp_code"]).get("induty_code", "")
    return {"as_of": as_of, "companies": list(companies.values())}


def _n_periods(corp_code: str) -> int:
    conn = sqlite3.connect(cf.DB_PATH)
    try:
        return conn.execute(
            "SELECT COUNT(DISTINCT bsns_year || reprt_code) FROM financials WHERE corp_code=?", (corp_code,)
        ).fetchone()[0]
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default="2026-09-23", help="시가총액 순위 기준일 (YYYY-MM-DD)")
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    snapshot = build_snapshot(args.as_of, args.top)
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"스냅샷 저장: {SNAPSHOT_PATH} ({len(snapshot['companies'])}개사)")

    for item in snapshot["companies"]:
        if _n_periods(item["corp_code"]) >= 4:
            print(f"[skip] {item['name']}: 이미 수집됨")
            continue
        try:
            n = cf.collect_recent_quarters(item["corp_code"])
            print(f"[collected] {item['name']}: 보고서 {n}개")
        except Exception as e:  # noqa: BLE001 — 한 회사가 실패해도 나머지는 계속 수집
            print(f"[error] {item['name']}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
