"""NAVER API HUB - 뉴스 검색 API 클라이언트.

Phase 2: (a) News Retriever Tool용 실시간 검색 호출,
         (b) 평가용 고정 코퍼스 수집 — 두 용도를 분리해서 사용 (Notion §1.5 참고)

## API 신청 (2026-07-31 이후: 개발자센터 신규 신청 차단, NCP API HUB로 이관됨)
1. https://www.ncloud.com 가입 (이메일 + 결제 카드 등록 필수 — 가입 승인 조건)
   → 가입 직후 Billing > 예산 알림을 낮은 금액(예: 100원)으로 설정해둘 것
2. 콘솔 > Menu > All Services > Application Services > NAVER API HUB
3. Subscription > 서비스 이용 신청
4. Application > Application 등록 > 검색 API 선택
5. Application Management > 해당 앱 > API 관리 > 인증 정보 > Client ID / Secret 확인

## 무료 한도 (2026-09 기준, NAVER API HUB FAQ)
- 일 25,000건 / 월 775,000건
- 한도 초과 시 "차단"되며 과금되지 않음 (50/80/100% 사용량 이메일 알림)
- 그래도 이 클라이언트는 네이버 한도보다 훨씬 낮은 자체 하드캡을 둔다 (버그로 인한
  폭주성 호출을 원천 차단하는 용도 — 이 프로젝트는 하루 수십~수백 건이면 충분함)
"""

import json
import os
from datetime import date
from pathlib import Path

import requests

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
BASE_URL = "https://openapi.naver.com/v1/search/news.json"  # TODO: NCP 콘솔에서 실제 엔드포인트 확인 후 수정

# 자체 안전장치: 네이버 무료 한도(일 25,000건)보다 훨씬 낮게 설정.
# 이 프로젝트가 실제로 필요한 호출량은 하루 수백 건 이내이므로 충분함.
DAILY_CALL_CAP = 300
_USAGE_FILE = Path(__file__).parent / ".news_api_usage.json"


class DailyCapExceeded(RuntimeError):
    """자체 설정한 일일 호출 한도를 넘었을 때 발생 (네이버 무료 한도 도달 훨씬 전에 멈춤)."""


def _load_usage() -> dict:
    if _USAGE_FILE.exists():
        return json.loads(_USAGE_FILE.read_text(encoding="utf-8"))
    return {}


def _bump_and_check_usage() -> int:
    today = date.today().isoformat()
    usage = _load_usage()
    count = usage.get(today, 0) + 1
    if count > DAILY_CALL_CAP:
        raise DailyCapExceeded(
            f"오늘({today}) 자체 설정한 일일 호출 한도({DAILY_CALL_CAP}건)를 초과했습니다. "
            "네이버 무료 한도(25,000건)와는 별개로, 버그로 인한 과호출을 막기 위한 자체 안전장치입니다. "
            "정말 더 호출해야 한다면 DAILY_CALL_CAP을 신중히 올리세요."
        )
    usage = {today: count}  # 날짜가 바뀌면 이전 기록은 버림 (단순화)
    _USAGE_FILE.write_text(json.dumps(usage), encoding="utf-8")
    return count


def search_news(query: str, display: int = 10, sort: str = "date") -> dict:
    """News Retriever Tool용 실시간 검색 호출.

    호출마다 자체 일일 한도를 체크한다 (DailyCapExceeded 발생 가능).
    """
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        raise RuntimeError("NAVER_CLIENT_ID / NAVER_CLIENT_SECRET이 .env에 설정되어 있지 않습니다.")

    _bump_and_check_usage()

    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    params = {"query": query, "display": display, "sort": sort}
    resp = requests.get(BASE_URL, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


# TODO: Phase 2에서 구현
# - collect_fixed_corpus(queries, start_date, end_date)  # 평가용 고정 수집 (별도 저장, search_news와 용도 분리)
