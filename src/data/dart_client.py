"""DART Open API 클라이언트.

Phase 1: 기업개황 / 재무제표(XBRL) / 공시 원문 조회.
API 신청: https://opendart.fss.or.kr (회원가입 → 인증키 신청/관리)
"""

import os

DART_API_KEY = os.getenv("DART_API_KEY")
BASE_URL = "https://opendart.fss.or.kr/api"

# TODO: Phase 1에서 구현
# - get_company_overview(corp_code)
# - get_financial_statement(corp_code, year, report_code)
# - search_disclosures(corp_code, start_date, end_date)
