"""네이버 뉴스 검색 API 클라이언트.

Phase 2: (a) News Retriever Tool용 실시간 검색 호출,
         (b) 평가용 고정 코퍼스 수집 — 두 용도를 분리해서 사용 (Notion §1.5 참고)
API 신청: https://developers.naver.com (Application 등록 → 검색 API)
"""

import os

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
BASE_URL = "https://openapi.naver.com/v1/search/news.json"

# TODO: Phase 2에서 구현
# - search_news(query, display=10, sort="date")  # 실시간 Tool 호출용
# - collect_fixed_corpus(queries, start_date, end_date)  # 평가용 고정 수집
