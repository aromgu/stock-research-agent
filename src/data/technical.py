"""주가 기술적 지표 계산 (이동평균/RSI/MACD/볼린저밴드).

순수 계산만 한다 (네트워크 호출 없음). 입력은 price_client.get_ohlcv()가 반환하는
형식(날짜 오름차순 list[dict], close/volume 포함)을 그대로 받는다.
"""

MA_WINDOWS = (5, 20, 60, 120, 240)
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BB_WINDOW, BB_STD = 20, 2
VOLUME_WINDOW = 20


def _closes(rows: list[dict]) -> list[float]:
    return [r["close"] for r in rows]


def _sma(values: list[float], window: int) -> float | None:
    if len(values) < window:
        return None
    return sum(values[-window:]) / window


def moving_averages(rows: list[dict], windows: tuple[int, ...] = MA_WINDOWS) -> dict[int, float | None]:
    """window일 -> 마지막 시점 기준 단순이동평균 (데이터가 window일보다 짧으면 None)."""
    closes = _closes(rows)
    return {w: _sma(closes, w) for w in windows}


def _ema_series(values: list[float], period: int) -> list[float]:
    """EMA 시계열. 첫 값은 앞 period개의 단순평균에서 시작 (표준적인 초기화 방식)."""
    if len(values) < period:
        return []
    k = 2 / (period + 1)
    ema = [sum(values[:period]) / period]
    for v in values[period:]:
        ema.append(v * k + ema[-1] * (1 - k))
    return ema


def rsi(rows: list[dict], period: int = RSI_PERIOD) -> float | None:
    """Wilder 방식 RSI. 마지막 시점 값 (데이터 부족하면 None)."""
    closes = _closes(rows)
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for g, loss in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        return 100.0
    rel_strength = avg_gain / avg_loss
    return 100 - 100 / (1 + rel_strength)


def rsi_zone(value: float) -> str:
    if value >= 70:
        return "과매수"
    if value <= 30:
        return "과매도"
    return "중립"


def macd(rows: list[dict]) -> dict | None:
    """MACD선/시그널선/히스토그램(마지막 값)과 최근 골든/데드크로스 여부. 데이터 부족하면 None."""
    closes = _closes(rows)
    ema_fast = _ema_series(closes, MACD_FAST)
    ema_slow = _ema_series(closes, MACD_SLOW)
    if not ema_fast or not ema_slow:
        return None
    offset = len(ema_fast) - len(ema_slow)  # slow가 더 늦게 시작하므로 겹치는 구간만 뺀다
    macd_line = [ema_fast[i + offset] - ema_slow[i] for i in range(len(ema_slow))]
    signal_line = _ema_series(macd_line, MACD_SIGNAL)
    if not signal_line:
        return None
    hist_offset = len(macd_line) - len(signal_line)
    histogram = [macd_line[i + hist_offset] - signal_line[i] for i in range(len(signal_line))]

    cross = None
    if len(histogram) >= 2:
        prev, cur = histogram[-2], histogram[-1]
        if prev <= 0 < cur:
            cross = "골든크로스(최근)"
        elif prev >= 0 > cur:
            cross = "데드크로스(최근)"

    return {"macd": macd_line[-1], "signal": signal_line[-1], "histogram": histogram[-1], "cross": cross}


def volume_vs_average(rows: list[dict], window: int = VOLUME_WINDOW) -> dict | None:
    """오늘 거래량 대비 직전 window일(오늘 제외) 평균 거래량 배율. 데이터 부족하면 None."""
    if len(rows) < window + 1:
        return None
    today = rows[-1]["volume"]
    baseline = rows[-window - 1 : -1]
    avg = sum(r["volume"] for r in baseline) / window
    if avg == 0:
        return None
    return {"today": today, "avg": avg, "ratio": today / avg}


def _band_at(closes: list[float], end_idx: int, window: int, n_std: float) -> tuple[float, float, float]:
    """closes[end_idx-window:end_idx] 구간의 (중심, 상단, 하단)."""
    window_vals = closes[end_idx - window : end_idx]
    mid = sum(window_vals) / window
    variance = sum((v - mid) ** 2 for v in window_vals) / window
    std = variance**0.5
    return mid, mid + n_std * std, mid - n_std * std


def bollinger(rows: list[dict], window: int = BB_WINDOW, n_std: float = BB_STD) -> dict | None:
    """마지막 시점 볼린저밴드 + 룰 기반 신호(밴드 이탈, 스퀴즈). 데이터 부족하면 None.

    스퀴즈는 예측이 아니라 관찰: 밴드폭이 최근 구간 중 최저치면 변동성이 축소된
    상태라는 뜻이고, 통계적으로 변동성 축소 후에는 확대가 뒤따르는 경우가 많다는
    경험칙을 그대로 문구로 노출한다.
    """
    closes = _closes(rows)
    if len(closes) < window:
        return None
    mid, upper, lower = _band_at(closes, len(closes), window, n_std)
    close = closes[-1]

    if close > upper:
        signal = "상단 돌파 — 단기 과열/강한 상승 추세 신호"
    elif close < lower:
        signal = "하단 이탈 — 단기 과매도/하락 추세 신호"
    else:
        signal = "밴드 내부 — 중립"

    width = (upper - lower) / mid * 100 if mid else None
    squeeze = None
    if width is not None and len(closes) >= window * 2:
        widths = []
        for i in range(window, len(closes) + 1):
            m, u, l = _band_at(closes, i, window, n_std)
            widths.append((u - l) / m * 100 if m else 0.0)
        if width <= min(widths):
            squeeze = "밴드폭이 최근 구간 중 최저 — 변동성 축소(스퀴즈), 방향성 전환 임박 가능성"

    return {
        "middle": mid,
        "upper": upper,
        "lower": lower,
        "close": close,
        "width_pct": width,
        "signal": signal,
        "squeeze": squeeze,
    }
