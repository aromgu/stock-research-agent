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

import html
import json
import os
import re
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests

_HTML_TAG_RE = re.compile(r"<[^>]+>")

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
# NAVER API HUB 이관 후 엔드포인트/헤더/도메인이 모두 바뀜 (기존 openapi.naver.com + X-Naver-Client-Id 방식 폐기).
BASE_URL = "https://naverapihub.apigw.ntruss.com/search/v1/news"

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


def _clean_text(raw: str) -> str:
    """NAVER 응답의 <b> 강조 태그와 HTML 엔티티(&quot; 등)를 제거."""
    return html.unescape(_HTML_TAG_RE.sub("", raw)).strip()


def search_news(query: str, display: int = 10, sort: str = "date") -> list[dict]:
    """News Retriever Tool용 실시간 검색 호출.

    호출마다 자체 일일 한도를 체크한다 (DailyCapExceeded 발생 가능).
    title/description은 HTML 태그를 제거한 순수 텍스트로 반환 (임베딩/재정렬에 바로 쓸 수 있도록).
    """
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        raise RuntimeError("NAVER_CLIENT_ID / NAVER_CLIENT_SECRET이 .env에 설정되어 있지 않습니다.")

    _bump_and_check_usage()

    headers = {
        "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
        "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET,
    }
    params = {"query": query, "display": display, "sort": sort}
    resp = requests.get(BASE_URL, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    return [
        {
            "title": _clean_text(item["title"]),
            "description": _clean_text(item["description"]),
            "link": item["link"],
            "originallink": item["originallink"],
            "pub_date": item["pubDate"],
        }
        for item in data.get("items", [])
    ]


_EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
_embedding_model = None  # 첫 호출 때만 로드 (약 2GB, DART만 쓰는 코드에선 불필요한 로딩 방지)


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(_EMBEDDING_MODEL_NAME)
    return _embedding_model


def embed_texts(texts: list[str]):
    """bge-m3로 텍스트 리스트를 정규화된 임베딩(코사인 유사도용)으로 변환. numpy.ndarray 반환.

    News Retriever Tool의 쿼리타임 재정렬과 평가용 고정 코퍼스 인덱싱(collect_news_corpus.py)이
    모델 로딩 로직을 공유하기 위한 공개 헬퍼.
    """
    model = _get_embedding_model()
    return model.encode(texts, normalize_embeddings=True)


def rerank_by_similarity(query: str, items: list[dict], top_k: int = 5) -> list[dict]:
    """title+description과 query의 임베딩 코사인 유사도로 재정렬 (상위 top_k만 반환).

    정적 인덱스 없이 검색 결과를 그 자리에서만 임베딩하는 쿼리타임 재정렬 (Notion §1.5).
    """
    if not items:
        return []

    texts = [f"{item['title']} {item['description']}" for item in items]
    embeddings = embed_texts([query] + texts)
    query_emb, item_embs = embeddings[0], embeddings[1:]
    scores = item_embs @ query_emb  # normalize_embeddings=True라 내적 = 코사인 유사도

    ranked = sorted(zip(items, scores), key=lambda pair: pair[1], reverse=True)
    return [{**item, "similarity_score": float(score)} for item, score in ranked[:top_k]]


def _parse_date_bound(value: str | None) -> datetime | None:
    """"YYYY-MM-DD" 문자열을 UTC 자정 기준 datetime으로. 날짜 단위 필터라 시간대 오차는 무시."""
    if value is None:
        return None
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def _filter_by_date_range(items: list[dict], after: datetime | None, before: datetime | None) -> list[dict]:
    result = []
    for item in items:
        pub = parsedate_to_datetime(item["pub_date"])
        if after is not None and pub < after:
            continue
        if before is not None and pub > before:
            continue
        result.append(item)
    return result


def _dedupe_similar_titles(items: list[dict], threshold: float = 0.85) -> list[dict]:
    """통신사 배포 기사 등 제목이 거의 동일한 중복을 제거 (먼저 나온 것만 남김).

    유사도만으로 top_k를 뽑으면 같은 사건을 베낀 기사가 슬롯을 다 차지해 다른 원인이
    있어도 못 보일 수 있어, 재정렬 전에 후보군에서부터 미리 다양성을 확보한다.
    """
    from difflib import SequenceMatcher

    kept = []
    for item in items:
        if any(SequenceMatcher(None, item["title"], k["title"]).ratio() >= threshold for k in kept):
            continue
        kept.append(item)
    return kept


def search_news_reranked(
    query: str,
    candidate_display: int = 20,
    top_k: int = 5,
    sort: str = "date",
    after: str | None = None,
    before: str | None = None,
    recency_days: int = 30,
) -> list[dict]:
    """News Retriever Tool 본체: 실시간 검색 -> 날짜 범위 필터 -> 중복 제거 -> 임베딩 재정렬.

    after/before("YYYY-MM-DD")를 지정하면 그 날짜 범위의 기사만 검색한다. 예: 2026년
    반기(4~6월) 실적 원인을 찾을 때는 "오늘 기준 최근 N일"이 아니라 그 분기 기간을 기준으로
    검색해야 한다 (원인이 된 사건이 오늘보다 훨씬 전, 분기 중간에 있을 수 있음).

    주의: NAVER 검색은 인기 검색어일수록 sort=date/sim 둘 다 최근 며칠치로 결과가 꽉 차서,
    display=100을 받아도 몇 달 전 과거로는 사실상 도달하지 못하는 경우가 흔하다(직접 확인함).
    그래서 after/before를 "명시적으로" 지정했는데 그 범위 안 기사가 하나도 없으면, 엉뚱한
    최신 기사로 조용히 바꿔치기하지 않고 빈 리스트를 그대로 반환한다 (호출부가 "그 기간
    기사를 못 찾았다"는 걸 알아야 하므로). after/before를 생략했을 때(기본 recency_days 모드)만
    검색어가 너무 좁아 후보가 부족하면 필터 없는 전체 후보로 관대하게 폴백한다.
    """
    candidates = search_news(query, display=candidate_display, sort=sort)
    explicit_range = bool(after or before)

    if explicit_range:
        after_dt, before_dt = _parse_date_bound(after), _parse_date_bound(before)
    else:
        after_dt, before_dt = datetime.now(timezone.utc) - timedelta(days=recency_days), None

    filtered = _filter_by_date_range(candidates, after_dt, before_dt)
    if explicit_range:
        pool = filtered  # 못 찾았으면 빈 채로 - 엉뚱한 기간 기사로 바꿔치기하지 않음
    else:
        pool = filtered if len(filtered) >= top_k else candidates
    deduped = _dedupe_similar_titles(pool)
    return rerank_by_similarity(query, deduped, top_k=top_k)

