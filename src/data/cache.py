"""외부 호출 결과 캐시 (로컬 SQLite: data/cache.db).

DART·NAVER·KRX 호출과 임베딩 계산은 같은 입력이면 같은 결과가 나오는 경우가 많다. 이미 가져온 것은
저장해두고 다시 쓰면 지연 시간과 외부 호출 수가 줄어든다. 값이 바뀔 수 있는 데이터(최근 주가, 최근 뉴스)는
TTL(유효 시간)을 짧게 주고, 바뀌지 않는 데이터(지난 날짜의 주가 등)는 TTL 없이 계속 쓴다.
"""

import hashlib
import json
import sqlite3
import time
from pathlib import Path

import numpy as np

CACHE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "cache.db"


def _connect() -> sqlite3.Connection:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # 도구를 병렬로 실행하면 여러 스레드가 동시에 쓰므로 잠금 대기 시간을 넉넉히 준다
    conn = sqlite3.connect(CACHE_PATH, timeout=30)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS kv (namespace TEXT, key TEXT, value TEXT, expires_at REAL, "
        "created_at REAL, PRIMARY KEY (namespace, key))"
    )
    conn.execute("CREATE TABLE IF NOT EXISTS embedding (key TEXT PRIMARY KEY, vec BLOB)")
    return conn


def make_key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def get(namespace: str, key: str):
    """캐시된 값(JSON으로 저장 가능한 값) 또는 None. 만료된 값은 없는 것으로 본다."""
    conn = _connect()
    try:
        row = conn.execute("SELECT value, expires_at FROM kv WHERE namespace=? AND key=?", (namespace, key)).fetchone()
    finally:
        conn.close()
    if row is None or (row[1] is not None and row[1] < time.time()):
        return None
    return json.loads(row[0])


def put(namespace: str, key: str, value, ttl_seconds: float | None = None) -> None:
    """값을 저장. ttl_seconds=None이면 만료 없음."""
    expires_at = time.time() + ttl_seconds if ttl_seconds is not None else None
    conn = _connect()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO kv VALUES (?, ?, ?, ?, ?)",
            (namespace, key, json.dumps(value, ensure_ascii=False), expires_at, time.time()),
        )
        conn.commit()
    finally:
        conn.close()


def get_embeddings(keys: list[str]) -> dict[str, np.ndarray]:
    conn = _connect()
    try:
        found = {}
        for i in range(0, len(keys), 500):  # SQLite 파라미터 개수 제한 때문에 나눠서 조회
            chunk = keys[i : i + 500]
            rows = conn.execute(
                f"SELECT key, vec FROM embedding WHERE key IN ({','.join('?' * len(chunk))})", chunk
            ).fetchall()
            found.update({k: np.frombuffer(v, dtype=np.float32) for k, v in rows})
        return found
    finally:
        conn.close()


def put_embeddings(items: dict[str, np.ndarray]) -> None:
    conn = _connect()
    try:
        conn.executemany(
            "INSERT OR REPLACE INTO embedding VALUES (?, ?)",
            [(k, np.asarray(v, dtype=np.float32).tobytes()) for k, v in items.items()],
        )
        conn.commit()
    finally:
        conn.close()
