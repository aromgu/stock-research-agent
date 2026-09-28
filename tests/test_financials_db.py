"""data/financials.db 자체의 무결성 점검 (재무제표 항등식) — API 호출 없음.

graders.py의 정답 계산과 tools.py의 조회 로직은 둘 다 이 DB를 그대로 읽는다. 두 코드가 같은
버그를 공유하면(예: 계정명 오타, fs_div 혼선) 정답과 조회 결과가 "똑같이 틀려서" 채점이 통과해
버릴 수 있다 — 이 테스트는 그 코드들을 거치지 않고 DB 원본에 회계 항등식(자산=부채+자본)만
직접 확인해, 저장 단계(collect_financials.py) 자체의 데이터 오류를 잡는다.

financials.db가 없는 환경(예: build_universe를 아직 안 돌린 새 clone)에서는 건너뛴다.
"""

import sqlite3
import unittest
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "financials.db"


def _amount(conn: sqlite3.Connection, corp_code: str, bsns_year: str, reprt_code: str, account_name: str) -> int | None:
    for fs_div in ("CFS", "OFS"):
        row = conn.execute(
            "SELECT thstrm_amount FROM financials WHERE corp_code=? AND bsns_year=? AND reprt_code=? "
            "AND account_name=? AND fs_div=? LIMIT 1",
            (corp_code, bsns_year, reprt_code, account_name, fs_div),
        ).fetchone()
        if row:
            return row[0]
    return None


def _latest_period(conn: sqlite3.Connection, corp_code: str) -> tuple[str, str] | None:
    row = conn.execute(
        "SELECT bsns_year, reprt_code FROM financials WHERE corp_code=? "
        "ORDER BY bsns_year DESC, collected_at DESC LIMIT 1",
        (corp_code,),
    ).fetchone()
    return tuple(row) if row else None


@unittest.skipUnless(DB_PATH.exists(), "financials.db가 없음 - 먼저 `python -m src.data.build_universe` 실행 필요")
class TestBalanceSheetIdentity(unittest.TestCase):
    """자산총계 = 부채총계 + 자본총계는 회계 항등식이라 DART 원본 데이터라면 항상 성립해야 한다.

    tools.py나 graders.py를 거치지 않고 DB를 직접 읽으므로, 두 코드가 같은 계정명·fs_div 버그를
    공유해도 이 테스트는 잡을 수 있다 (저장 단계 자체의 문제인지 구분하는 용도).
    """

    # 삼성전자, SK하이닉스 corp_code (DART 고유 8자리)
    _COMPANIES = {"00126380": "삼성전자", "00164779": "SK하이닉스"}

    def test_assets_equal_liabilities_plus_equity(self):
        conn = sqlite3.connect(DB_PATH)
        try:
            checked = 0
            for corp_code, name in self._COMPANIES.items():
                period = _latest_period(conn, corp_code)
                if period is None:
                    continue
                year, reprt_code = period
                assets = _amount(conn, corp_code, year, reprt_code, "자산총계")
                liabilities = _amount(conn, corp_code, year, reprt_code, "부채총계")
                equity = _amount(conn, corp_code, year, reprt_code, "자본총계")
                if assets is None or liabilities is None or equity is None:
                    continue
                checked += 1
                diff_ratio = abs(assets - (liabilities + equity)) / max(abs(assets), 1)
                self.assertLess(
                    diff_ratio, 0.001,
                    f"{name} {year}년 {reprt_code}: 자산총계({assets:,}) != 부채총계+자본총계({liabilities + equity:,})",
                )
            if checked == 0:
                self.skipTest("DB에 대상 회사의 재무상태표 데이터가 없음 (아직 수집 전)")
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
