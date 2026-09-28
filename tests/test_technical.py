"""기술적 지표 계산 단위 테스트 (순수 계산, 네트워크 없음). 실행: python -m unittest discover tests"""

import unittest

from src.data import technical as tech


def _rows(closes: list[float], volumes: list[int] | None = None) -> list[dict]:
    volumes = volumes or [1000] * len(closes)
    return [
        {"date": f"2026-01-{i + 1:02d}", "close": c, "volume": v} for i, (c, v) in enumerate(zip(closes, volumes))
    ]


class TestMovingAverages(unittest.TestCase):
    def test_sma_matches_manual_average(self):
        rows = _rows([10, 20, 30, 40, 50])
        result = tech.moving_averages(rows, windows=(3, 5))
        self.assertAlmostEqual(result[3], (30 + 40 + 50) / 3)
        self.assertAlmostEqual(result[5], (10 + 20 + 30 + 40 + 50) / 5)

    def test_insufficient_data_is_none(self):
        rows = _rows([10, 20])
        result = tech.moving_averages(rows, windows=(5,))
        self.assertIsNone(result[5])


class TestRsi(unittest.TestCase):
    def test_monotonic_increase_is_100(self):
        rows = _rows(list(range(1, 20)))  # 계속 오르기만 하면 손실이 없어 RSI 100
        self.assertEqual(tech.rsi(rows), 100.0)

    def test_monotonic_decrease_is_0(self):
        rows = _rows(list(range(20, 1, -1)))
        self.assertEqual(tech.rsi(rows), 0.0)

    def test_insufficient_data_is_none(self):
        self.assertIsNone(tech.rsi(_rows([1, 2, 3])))

    def test_zone_labels(self):
        self.assertEqual(tech.rsi_zone(80), "과매수")
        self.assertEqual(tech.rsi_zone(20), "과매도")
        self.assertEqual(tech.rsi_zone(50), "중립")


class TestMacd(unittest.TestCase):
    def test_insufficient_data_is_none(self):
        self.assertIsNone(tech.macd(_rows([1, 2, 3])))

    def test_uptrend_has_positive_histogram_eventually(self):
        rows = _rows([100 + i for i in range(60)])  # 꾸준한 상승 추세
        result = tech.macd(rows)
        self.assertIsNotNone(result)
        self.assertGreater(result["macd"], 0)  # 상승 추세에서는 단기EMA가 장기EMA보다 위


class TestVolumeVsAverage(unittest.TestCase):
    def test_ratio_excludes_today_from_baseline(self):
        volumes = [100] * 20 + [400]  # 오늘만 4배 급증
        rows = _rows(list(range(21)), volumes)
        result = tech.volume_vs_average(rows, window=20)
        self.assertAlmostEqual(result["avg"], 100)
        self.assertAlmostEqual(result["ratio"], 4.0)

    def test_insufficient_data_is_none(self):
        self.assertIsNone(tech.volume_vs_average(_rows([1, 2]), window=20))


class TestBollinger(unittest.TestCase):
    def test_constant_price_gives_zero_width_and_neutral(self):
        rows = _rows([100] * 20)
        result = tech.bollinger(rows, window=20)
        self.assertAlmostEqual(result["upper"], 100)
        self.assertAlmostEqual(result["lower"], 100)
        self.assertEqual(result["signal"], "밴드 내부 — 중립")

    def test_spike_breaks_upper_band(self):
        rows = _rows([100] * 19 + [200])
        result = tech.bollinger(rows, window=20)
        self.assertGreater(result["close"], result["upper"])
        self.assertIn("상단 돌파", result["signal"])

    def test_insufficient_data_is_none(self):
        self.assertIsNone(tech.bollinger(_rows([1, 2, 3]), window=20))


if __name__ == "__main__":
    unittest.main()
