"""graders._peer_metric_independent()가 tools._peer_metric()과 독립적으로 짠 코드인데도 같은 값을
내는지 확인 (API 호출 없음, financials.db/universe_snapshot.json 필요).

이 둘이 코드가 다른데 계속 일치한다는 것 자체가 "순환 채점"(같은 버그를 공유해서 우연히 맞아떨어지는 것)이
아니라는 근거가 된다 — 둘 중 하나만 고쳐도 이 테스트가 잡아낸다.
"""

import unittest
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "financials.db"
SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "data" / "universe_snapshot.json"


@unittest.skipUnless(
    DB_PATH.exists() and SNAPSHOT_PATH.exists(),
    "financials.db/universe_snapshot.json이 없음 - 먼저 `python -m src.data.build_universe` 실행 필요",
)
class TestPeerMetricIndependentMatchesTool(unittest.TestCase):
    def test_ratio_and_amount_metrics_agree_for_real_companies(self):
        from src.agent import tools
        from src.data import universe as uv
        from src.eval import graders as g

        companies = uv.load_universe()[:10]  # 스냅샷 앞 10개사면 충분 (전체를 다 돌 필요 없음)
        metrics = ["영업이익률", "순이익률", "부채비율", "매출액 증가율", "매출액"]

        checked = 0
        for c in companies:
            for metric in metrics:
                expected, _ = tools._peer_metric(c["corp_code"], metric)
                actual, _ = g._peer_metric_independent(c["corp_code"], metric)
                if expected is None and actual is None:
                    continue
                checked += 1
                self.assertIsNotNone(actual, f"{c['name']} {metric}: tools={expected}, graders=None")
                self.assertIsNotNone(expected, f"{c['name']} {metric}: tools=None, graders={actual}")
                self.assertAlmostEqual(
                    expected, actual, places=2,
                    msg=f"{c['name']} {metric}: tools={expected} vs graders(독립 계산)={actual}",
                )
        self.assertGreater(checked, 0, "비교할 수 있는 (회사, 지표) 조합이 하나도 없음")


if __name__ == "__main__":
    unittest.main()
