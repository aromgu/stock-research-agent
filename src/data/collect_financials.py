"""Phase 1: DART 재무 데이터를 SQLite에 수집/저장.

기업별 최근 N개 보고서(분기/반기/연간)를 받아 로컬 DB(data/financials.db)에 저장한다.
매번 라이브 API를 호출하지 않고 이 DB를 채점/분석용 데이터 소스로 재사용하기 위함.
"""

import sqlite3
from datetime import date, datetime
from pathlib import Path

from . import dart_client as dc

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "financials.db"

# 반도체 산업 실습 대상 (Notion 계획서 §2.3 스코프 결정에 따름)
TARGET_COMPANIES = ["005930", "000660"]  # 삼성전자, SK하이닉스

# (reprt_code, 분기 말월) - 연도 내 오름차순. 11011=사업보고서 연간분은 12월로 취급.
_PERIODS_IN_YEAR = [("11013", 3), ("11012", 6), ("11014", 9), ("11011", 12)]


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS financials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            corp_code TEXT NOT NULL,
            corp_name TEXT NOT NULL,
            stock_code TEXT NOT NULL,
            bsns_year TEXT NOT NULL,
            reprt_code TEXT NOT NULL,
            fs_div TEXT NOT NULL,
            fs_name TEXT NOT NULL,
            sj_name TEXT NOT NULL,
            account_name TEXT NOT NULL,
            thstrm_amount INTEGER,
            frmtrm_amount INTEGER,
            bfefrmtrm_amount INTEGER,
            thstrm_add_amount INTEGER,
            frmtrm_add_amount INTEGER,
            collected_at TEXT NOT NULL,
            UNIQUE(corp_code, bsns_year, reprt_code, fs_div, sj_name, account_name)
        )
        """
    )
    conn.commit()


def _recent_period_candidates(as_of: date, n_candidates: int = 8) -> list[tuple[str, str]]:
    """as_of 기준 최근 분기부터 역순으로 (year, reprt_code) 후보를 만든다.

    실제로 공시가 올라왔는지는 모르니 "후보"일 뿐 — 호출하는 쪽에서 status=013(미공시)을
    건너뛰며 실제로 존재하는 보고서만 골라낸다.
    """
    candidates = []
    year = as_of.year
    while len(candidates) < n_candidates:
        for reprt_code, end_month in reversed(_PERIODS_IN_YEAR):
            if year == as_of.year and end_month > as_of.month:
                continue  # 아직 끝나지 않은 분기는 후보에서 제외
            candidates.append((str(year), reprt_code))
            if len(candidates) >= n_candidates:
                break
        year -= 1
    return candidates


def _store_report(conn: sqlite3.Connection, corp_code: str, overview: dict, year: str, reprt_code: str) -> bool:
    """보고서 하나를 DART에서 받아 DB에 저장. 아직 공시 안 된 보고서(status=013)면 False."""
    try:
        rows = dc.get_financial_statement(corp_code, year, reprt_code)
    except dc.DartApiError as e:
        if e.status == "013":  # 아직 공시 안 됨 - 정상 상황
            return False
        raise

    now = datetime.now().isoformat(timespec="seconds")
    conn.executemany(
        """
        INSERT OR REPLACE INTO financials
            (corp_code, corp_name, stock_code, bsns_year, reprt_code, fs_div,
             fs_name, sj_name, account_name, thstrm_amount, frmtrm_amount, bfefrmtrm_amount,
             thstrm_add_amount, frmtrm_add_amount, collected_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                corp_code,
                overview["corp_name"],
                overview["stock_code"],
                year,
                reprt_code,
                row["fs_div"],
                row["fs_name"],
                row["sj_name"],
                row["account_name"],
                row["thstrm_amount"],
                row["frmtrm_amount"],
                row["bfefrmtrm_amount"],
                row["thstrm_add_amount"],
                row["frmtrm_add_amount"],
                now,
            )
            for row in rows
        ],
    )
    conn.commit()
    return True


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    _init_db(conn)
    return conn


def collect_recent_quarters(name_or_stock_code: str, n_quarters: int = 4) -> int:
    """기업의 최근 n_quarters개 보고서를 DART에서 받아 로컬 DB에 저장.

    아직 공시되지 않은 분기(DART status=013)는 조용히 건너뛰고 다음 후보로 넘어간다.
    그 외 API 에러는 그대로 올린다. 실제로 저장한 보고서 개수를 반환.
    """
    corp_code = dc.get_corp_code(name_or_stock_code)
    overview = dc.get_company_overview(corp_code)
    conn = _connect()
    collected = 0
    try:
        for year, reprt_code in _recent_period_candidates(date.today()):
            if collected >= n_quarters:
                break
            if _store_report(conn, corp_code, overview, year, reprt_code):
                collected += 1
    finally:
        conn.close()
    return collected


def fetch_report(name_or_stock_code: str, year: str, reprt_code: str) -> bool:
    """특정 보고서 하나를 DART에서 받아 저장 (도구가 DB에 없는 기간을 요청받았을 때 사용)."""
    corp_code = dc.get_corp_code(name_or_stock_code)
    overview = dc.get_company_overview(corp_code)
    conn = _connect()
    try:
        return _store_report(conn, corp_code, overview, str(year), reprt_code)
    finally:
        conn.close()


if __name__ == "__main__":
    for company in TARGET_COMPANIES:
        n = collect_recent_quarters(company)
        print(f"{company}: {n}개 보고서 수집 완료 -> {DB_PATH}")
