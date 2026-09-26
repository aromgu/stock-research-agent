"""주가/밸류에이션 클라이언트 (pykrx — 한국거래소 KRX 데이터, API 키 불필요).

- 일별 시세(시가/고가/저가/종가/거래량): 로그인 없이 조회됨
- PER/PBR/EPS/BPS/배당수익률, 시가총액: KRX 정보데이터시스템(data.krx.co.kr) 로그인이
  필요해졌다 (2026-09 확인, 로그인 없이 호출하면 빈 DataFrame). .env에 KRX_ID/KRX_PW 필요.
"""

from datetime import date, timedelta

from dotenv import load_dotenv

load_dotenv()  # pykrx가 import 시점에 KRX_ID/KRX_PW로 로그인을 시도하므로 import보다 먼저 로드해야 함

from pykrx import stock  # noqa: E402

from . import cache  # noqa: E402

# 조회일이 휴장일일 수 있어서, 그 날짜 이전 며칠까지 범위로 받아 마지막 거래일 값을 쓴다
_LOOKBACK_DAYS = 10
_RECENT_TTL_SECONDS = 600  # 오늘이 포함된 조회는 장중에 값이 바뀌므로 10분만 재사용


def _krx_date(d: str) -> str:
    return d.replace("-", "")


def _cached(namespace: str, args: tuple, end: str, fetch):
    """지난 날짜까지의 시세는 바뀌지 않으므로 만료 없이 캐시, 오늘이 포함되면 짧게 캐시.

    빈 결과는 캐시하지 않는다 — 로그인 실패나 일시적 오류로 빈 값이 영구 저장되는 것을 막기 위해.
    """
    key = cache.make_key(*args)
    hit = cache.get(namespace, key)
    if hit is not None:
        return hit
    value = fetch()
    if value:
        ttl = None if end < date.today().isoformat() else _RECENT_TTL_SECONDS
        cache.put(namespace, key, value, ttl)
    return value


def get_ohlcv(ticker: str, start: str, end: str) -> list[dict]:
    """start~end(YYYY-MM-DD) 일별 시세. 거래일만 반환 (휴장일 제외)."""
    return _cached("krx_ohlcv", (ticker, start, end), end, lambda: _fetch_ohlcv(ticker, start, end))


def get_fundamental(ticker: str, on: str) -> dict | None:
    """on(YYYY-MM-DD) 기준 직전 거래일의 PER/PBR/EPS/BPS/DIV. KRX 로그인 없으면 None."""
    return _cached("krx_fundamental", (ticker, on), on, lambda: _fetch_fundamental(ticker, on))


def get_market_cap(ticker: str, on: str) -> dict | None:
    """on(YYYY-MM-DD) 기준 직전 거래일의 시가총액/상장주식수. KRX 로그인 없으면 None."""
    return _cached("krx_market_cap", (ticker, on), on, lambda: _fetch_market_cap(ticker, on))


def _fetch_ohlcv(ticker: str, start: str, end: str) -> list[dict]:
    df = stock.get_market_ohlcv(_krx_date(start), _krx_date(end), ticker)
    return [
        {
            "date": idx.strftime("%Y-%m-%d"),
            "open": int(row["시가"]),
            "high": int(row["고가"]),
            "low": int(row["저가"]),
            "close": int(row["종가"]),
            "volume": int(row["거래량"]),
        }
        for idx, row in df.iterrows()
    ]


def _lookback_start(on: str) -> str:
    return (date.fromisoformat(on) - timedelta(days=_LOOKBACK_DAYS)).isoformat()


def _fetch_fundamental(ticker: str, on: str) -> dict | None:
    df = stock.get_market_fundamental(_krx_date(_lookback_start(on)), _krx_date(on), ticker)
    if df.empty:
        return None
    row = df.iloc[-1]
    return {
        "date": df.index[-1].strftime("%Y-%m-%d"),
        "PER": float(row["PER"]),
        "PBR": float(row["PBR"]),
        "EPS": float(row["EPS"]),
        "BPS": float(row["BPS"]),
        "DIV": float(row["DIV"]),
    }


def _fetch_market_cap(ticker: str, on: str) -> dict | None:
    df = stock.get_market_cap(_krx_date(_lookback_start(on)), _krx_date(on), ticker)
    if df.empty:
        return None
    row = df.iloc[-1]
    return {
        "date": df.index[-1].strftime("%Y-%m-%d"),
        "market_cap": int(row["시가총액"]),
        "shares": int(row["상장주식수"]),
    }
