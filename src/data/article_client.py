"""뉴스 기사 본문 수집 (에이전트가 필요할 때만 부르는 read_articles 도구용).

검색 결과의 요약문(description)은 한두 문장이라 "영업이익이 왜 늘었는지" 같은 구체적 원인이 빠질 때가 많다.
그렇다고 매 검색마다 본문을 다 가져오면 느리고 토큰도 많이 쓰므로, 에이전트가 요약만으로 부족하다고 판단한
기사만 가져온다.

사이트마다 크롤러를 따로 두지 않는다: NAVER 뉴스 페이지(n.news.naver.com)는 언론사와 무관하게 구조가 같아
파서 하나로 되고, 나머지 사이트는 공통 규칙(<article> 또는 가장 긴 문단 묶음)으로 뽑는다. 한 번 가져온
본문은 바뀌지 않으므로 캐시에 영구 저장한다.
"""

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor

import requests
from bs4 import BeautifulSoup

from . import cache

_HEADERS = {"User-Agent": "Mozilla/5.0 (stock-research-agent; personal research)"}
_TIMEOUT = 6
_MAX_CHARS = 1500  # 기사 하나당 에이전트에 넘길 최대 길이 (토큰 절약 — 원인 설명은 보통 앞부분에 있음)
_MIN_BODY_CHARS = 200  # 이보다 짧으면 본문을 못 뽑은 것으로 봄 (메뉴·광고 문구만 잡힌 경우)


def article_id(link: str) -> str:
    """검색 결과에 붙여 보여줄 짧은 기사 ID. 링크 전체보다 토큰이 훨씬 적다."""
    return hashlib.sha1(link.encode("utf-8")).hexdigest()[:8]


def remember_links(items: list[dict]) -> None:
    """검색 결과의 기사 ID → 링크를 저장해 두고 read_articles가 ID로 찾게 한다."""
    for item in items:
        cache.put("article_link", article_id(item["link"]), {"link": item["link"], "title": item["title"]})


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_naver(soup: BeautifulSoup) -> str | None:
    node = soup.select_one("#dic_area") or soup.select_one("#newsct_article")
    return _clean(node.get_text(" ")) if node else None


def _extract_generic(soup: BeautifulSoup) -> str | None:
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()
    article = soup.find("article")
    if article:
        text = _clean(article.get_text(" "))
        if len(text) >= _MIN_BODY_CHARS:
            return text
    # 글이 직접(태그 없이 <br>로 나눠서, 또는 바로 아래 <p>로) 가장 많이 들어 있는 블록을 본문으로 본다.
    # "<p>가 가장 많은 블록"만 보면 본문을 <br>로 쓰는 사이트에서 사업자번호 같은 하단 정보를 본문으로 잡았다.
    best, best_len = None, 0
    for block in soup.find_all(["div", "section", "td"]):
        direct = " ".join(s for s in block.find_all(string=True, recursive=False))
        paragraphs = " ".join(p.get_text(" ") for p in block.find_all("p", recursive=False))
        text = _clean(f"{direct} {paragraphs}")
        if len(text) > best_len:
            best, best_len = text, len(text)
    if best_len >= _MIN_BODY_CHARS:
        return best
    meta = soup.find("meta", property="og:description")  # 본문을 못 찾으면 언론사가 넣어 둔 요약이라도
    return _clean(meta["content"]) if meta and meta.get("content") else None


def fetch_article_text(link: str) -> str | None:
    """기사 본문. 못 가져오면 None (사이트 차단·구조 차이 등). 성공한 본문만 영구 캐시."""
    key = cache.make_key(link)
    hit = cache.get("article_text", key)
    if hit is not None:
        return hit
    try:
        resp = requests.get(link, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException:
        return None
    if not resp.encoding or resp.encoding.lower() == "iso-8859-1":
        resp.encoding = resp.apparent_encoding  # charset 헤더가 없는 한국 사이트가 있어 글자가 깨지지 않게
    soup = BeautifulSoup(resp.text, "lxml")
    text = _extract_naver(soup) if "n.news.naver.com" in link else _extract_generic(soup)
    if not text or len(text) < _MIN_BODY_CHARS:
        return None
    cache.put("article_text", key, text)
    return text


def read_articles(ids: list[str]) -> list[dict]:
    """기사 ID들의 본문을 동시에 가져온다. 반환: [{"id", "title", "text"(없으면 None), "error"(있으면)}]"""
    entries = []
    for aid in ids[:5]:
        meta = cache.get("article_link", aid.strip())
        entries.append({"id": aid, **(meta or {})})

    def load(entry: dict) -> dict:
        if "link" not in entry:
            return {**entry, "text": None, "error": "알 수 없는 기사 ID (search_news 결과의 id를 그대로 써야 함)"}
        text = fetch_article_text(entry["link"])
        if text is None:
            return {**entry, "text": None, "error": "본문을 가져오지 못함 (사이트 차단 또는 구조 문제)"}
        return {**entry, "text": text[:_MAX_CHARS] + ("…" if len(text) > _MAX_CHARS else "")}

    with ThreadPoolExecutor(max_workers=5) as pool:
        return list(pool.map(load, entries))
