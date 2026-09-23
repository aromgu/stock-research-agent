"""Phase 2: 평가용 고정 뉴스 코퍼스 수집 (Chroma 벡터 DB).

News Retriever Tool(news_client.search_news_reranked)은 매번 실시간으로 API를 호출하지만,
채점의 재현성을 위해서는 "이 시점 기준 코퍼스"가 고정돼 있어야 한다 (Notion §1.5, §2.3).
이 스크립트는 특정 시점에 여러 쿼리로 뉴스를 모아 임베딩째 Chroma에 저장해둔다.
"""

import hashlib
from datetime import datetime
from pathlib import Path

import chromadb

from . import news_client as nc

CHROMA_PATH = Path(__file__).resolve().parent.parent.parent / "chroma"
COLLECTION_NAME = "news_corpus_fixed"
EXPORT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "news_corpus_export.md"

_STOCK_NAMES = {"005930": "삼성전자", "000660": "SK하이닉스"}

# 주의: 이 컬렉션은 embedding_function 없이 news_client.embed_texts()(bge-m3, 1024차원)로
# 미리 계산한 벡터를 upsert한다. 나중에 이 컬렉션을 조회할 때 collection.query(query_texts=...)를
# 쓰면 Chroma 기본 임베딩 함수(all-MiniLM-L6-v2, 384차원)가 대신 쓰여 차원 불일치 에러가 난다.
# 반드시 nc.embed_texts(...)로 쿼리 임베딩을 만들어 query_embeddings=...로 넘길 것.

# 반도체 산업 실습 대상 (Notion 계획서 §2.3 스코프와 동일: 삼성전자, SK하이닉스)
CORPUS_QUERIES = {
    "005930": ["삼성전자 실적", "삼성전자 반도체", "삼성전자 메모리"],
    "000660": ["SK하이닉스 실적", "SK하이닉스 반도체", "SK하이닉스 HBM"],
}


def _make_id(link: str) -> str:
    return hashlib.sha1(link.encode("utf-8")).hexdigest()


def collect_fixed_corpus(queries: dict[str, list[str]] = CORPUS_QUERIES, display_per_query: int = 30) -> int:
    """쿼리별로 최신 뉴스를 모아 링크 기준 중복 제거 후 Chroma에 저장.

    같은 링크가 이미 있으면 upsert로 덮어써 재실행해도 중복 저장되지 않는다.
    저장한(신규+갱신) 문서 개수를 반환.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    seen_links = {}
    for stock_code, query_list in queries.items():
        for query in query_list:
            for item in nc.search_news(query, display=display_per_query, sort="date"):
                seen_links[item["link"]] = (stock_code, query, item)

    if not seen_links:
        return 0

    ids, documents, metadatas = [], [], []
    now = datetime.now().isoformat(timespec="seconds")
    for link, (stock_code, query, item) in seen_links.items():
        ids.append(_make_id(link))
        documents.append(f"{item['title']} {item['description']}")
        metadatas.append(
            {
                "stock_code": stock_code,
                "query": query,
                "title": item["title"],
                "description": item["description"],
                "link": link,
                "originallink": item["originallink"],
                "pub_date": item["pub_date"],
                "collected_at": now,
            }
        )

    embeddings = nc.embed_texts(documents).tolist()
    collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
    return len(ids)


def export_corpus_to_markdown(export_path: Path = EXPORT_PATH) -> int:
    """Chroma에 저장된 코퍼스를 사람이 직접 훑어볼 수 있는 마크다운 파일로 내보낸다."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(COLLECTION_NAME)
    data = collection.get(include=["metadatas"])
    metas = sorted(data["metadatas"], key=lambda m: (m["stock_code"], m["pub_date"]), reverse=True)

    lines = [f"# 뉴스 코퍼스 (fixed, Chroma `{COLLECTION_NAME}`)", "", f"총 {len(metas)}건 수집", ""]
    for meta in metas:
        stock_name = _STOCK_NAMES.get(meta["stock_code"], meta["stock_code"])
        lines.append(f"## [{stock_name}] {meta['title']}")
        lines.append(f"- 검색쿼리: {meta['query']}")
        lines.append(f"- 발행일: {meta['pub_date']}")
        lines.append(f"- 링크: {meta['link']}")
        lines.append(f"- 요약: {meta['description']}")
        lines.append("")

    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_text("\n".join(lines), encoding="utf-8")
    return len(metas)


if __name__ == "__main__":
    n = collect_fixed_corpus()
    print(f"{n}개 기사 수집 완료 -> {CHROMA_PATH}")
    n_exported = export_corpus_to_markdown()
    print(f"{n_exported}개 기사 export 완료 -> {EXPORT_PATH}")
