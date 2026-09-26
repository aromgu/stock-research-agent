"""지원 종목 50개의 뉴스를 미리 모아 아카이브(data/cache.db)에 쌓는 배치 수집기.

NAVER 검색은 최근 1,000건까지만 보여줘서, 기사가 많은 종목은 몇 달만 지나도 그 시기 기사에 닿지 못한다.
질문이 들어올 때만 검색하면 아카이브에 빈틈이 생기므로, 매일(또는 며칠마다) 돌려 두면 과거 기간 질문에서도
아카이브가 NAVER 대신 기사를 채워 준다.

NAVER 호출을 아끼기 위해 최신순으로 넘기다가, 한 페이지가 전부 이미 가진 기사면 거기서 멈춘다
(매일 돌리면 종목·검색어당 보통 1~2번 호출로 끝난다).

실행: python -m src.data.collect_news [--pages 10] [--budget 3000] [--embed-days 7]
"""

import argparse
import sqlite3
import time

from . import news_client as nc
from .universe import load_universe

_QUERY_TEMPLATES = ("{name}", "{name} 실적")  # 회사 전반 + 실적 원인 기사 (실적 질문이 가장 많음)


def collect_query(query: str, max_pages: int) -> tuple[int, int]:
    """(NAVER 호출 수, 새로 쌓은 기사 수)"""
    calls = new = 0
    for page in range(max_pages):
        items = nc.search_news(query, display=100, sort="date", start=page * 100 + 1)  # 호출 시 아카이브에 저장됨
        calls += 1
        if not items:
            break
        # search_news가 호출 즉시 아카이브에 넣으므로, '이번 실행 중에 처음 저장된 기사' 수로 새 기사를 센다
        fresh = _fresh_count([i["link"] for i in items])
        new += fresh
        if fresh == 0 or len(items) < 100:
            break
    return calls, new


_run_started = time.time()


def _fresh_count(links: list[str]) -> int:
    conn = nc._archive_connect()
    try:
        placeholders = ",".join("?" * len(links))
        (n,) = conn.execute(
            f"SELECT COUNT(*) FROM news_archive WHERE link IN ({placeholders}) AND fetched_at >= ?",
            (*links, _run_started),
        ).fetchone()
    finally:
        conn.close()
    return n


def precompute_embeddings(days: int) -> int:
    """최근 N일 아카이브 기사를 미리 임베딩해 둔다 (검색 때는 질문 문장만 임베딩하면 되도록).

    CPU에서 기사당 약 0.2초라 검색 한 번의 재정렬(최대 80건)이 10초 넘게 걸리던 것을 수집 시점으로 옮긴다.
    재정렬과 같은 문장 형식("제목 요약")을 써야 캐시가 맞는다. 이미 임베딩된 기사는 건너뛴다.
    """
    since = time.time() - days * 86400
    conn = nc._archive_connect()
    try:
        rows = conn.execute("SELECT title, description FROM news_archive WHERE pub_ts >= ?", (since,)).fetchall()
    finally:
        conn.close()
    texts = [f"{t} {d}" for t, d in rows]
    for i in range(0, len(texts), 256):
        nc.embed_texts(texts[i : i + 256])
        print(f"임베딩 {min(i + 256, len(texts)):,}/{len(texts):,}", flush=True)
    return len(texts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=10, help="검색어당 최대 페이지 (100건씩, NAVER 최대 10)")
    parser.add_argument("--budget", type=int, default=3000, help="이번 실행의 NAVER 호출 상한 (일 한도 25,000)")
    parser.add_argument("--embed-days", type=int, default=0, help="최근 N일 기사를 미리 임베딩 (0이면 안 함, CPU 기사당 약 0.2초)")
    parser.add_argument("--skip-collect", action="store_true", help="수집 없이 임베딩만")
    args = parser.parse_args()

    conn = nc._archive_connect()
    (before,) = conn.execute("SELECT COUNT(*) FROM news_archive").fetchone()
    conn.close()

    total_calls = total_new = 0
    start = time.perf_counter()
    for company in [] if args.skip_collect else load_universe():
        for template in _QUERY_TEMPLATES:
            if total_calls + args.pages > args.budget:
                print(f"호출 상한 {args.budget}에 가까워 중단")
                break
            query = template.format(name=company["name"])
            try:
                calls, new = collect_query(query, args.pages)
            except (nc.DailyCapExceeded, sqlite3.Error, OSError) as e:
                print(f"{query}: 중단 ({e})")
                continue
            total_calls += calls
            total_new += new
            print(f"{query}: 호출 {calls}번, 새 기사 {new}건", flush=True)

    conn = nc._archive_connect()
    (after,) = conn.execute("SELECT COUNT(*) FROM news_archive").fetchone()
    conn.close()
    print(
        f"\n완료: NAVER 호출 {total_calls}번, {time.perf_counter() - start:.0f}초. "
        f"아카이브 {before:,}건 → {after:,}건"
    )
    if args.embed_days:
        start = time.perf_counter()
        n = precompute_embeddings(args.embed_days)
        print(f"최근 {args.embed_days}일 기사 {n:,}건 임베딩 확인, {time.perf_counter() - start:.0f}초")


if __name__ == "__main__":
    main()
