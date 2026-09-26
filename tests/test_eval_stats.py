"""평가 통계·held-out v4 고정 단위 테스트 (API 호출 없음)."""

import hashlib
import json
import unittest
from pathlib import Path

from src.eval import stats


class TestBootstrap(unittest.TestCase):
    def test_small_sample_interval_contains_zero(self):
        # held-out v3 상황: 6문항 중 한 문항만 차이 → 구간이 0을 포함해야 함
        mean, lo, hi = stats.paired_bootstrap_ci([1, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(mean, 1 / 6)
        self.assertLessEqual(lo, 0)
        self.assertGreater(hi, 0)

    def test_consistent_large_difference_excludes_zero(self):
        mean, lo, hi = stats.paired_bootstrap_ci([1] * 30 + [0] * 15)
        self.assertGreater(lo, 0)

    def test_deterministic(self):
        diffs = [1, -1, 0, 1, 0.5]
        self.assertEqual(stats.paired_bootstrap_ci(diffs), stats.paired_bootstrap_ci(diffs))


class TestRepeatAggregation(unittest.TestCase):
    ROWS = [
        {"question_id": "a", "approach": "agent", "judge_pass": True},
        {"question_id": "a", "approach": "agent", "judge_pass": False},
        {"question_id": "a", "approach": "agent", "judge_pass": True},
        {"question_id": "b", "approach": "agent", "judge_pass": True},
        {"question_id": "b", "approach": "agent", "judge_pass": True},
        {"question_id": "c", "approach": "agent", "judge_pass": None},  # 판정 불가는 빼고 셈
    ]

    def test_rate_and_flips(self):
        self.assertEqual(stats.per_question_rate(self.ROWS, "agent"), {"a": 2 / 3, "b": 1.0})
        self.assertEqual(stats.flip_count(self.ROWS, "agent"), (1, 2))


class TestHeldoutV4Frozen(unittest.TestCase):
    def test_file_matches_frozen_hash_and_is_balanced(self):
        from src.eval.gold_set import HELDOUT_V4_QUESTIONS, HELDOUT_V4_SHA256

        path = Path(__file__).resolve().parent.parent / "src" / "eval" / "heldout_v4.json"
        raw = path.read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(hashlib.sha256(raw).hexdigest(), HELDOUT_V4_SHA256)
        self.assertEqual(len(HELDOUT_V4_QUESTIONS), 45)
        # 개발 중 보던 두 회사는 v4 대상에서 제외
        payload = json.loads(raw)
        self.assertFalse(any(q["company"] in ("삼성전자", "SK하이닉스") for q in payload["questions"]))
        # 전제 문항은 맞는 전제와 틀린 전제가 섞여 있어야 함 (루브릭의 "실제로는" 방향과 질문의 방향 비교)
        premise = [q for q in payload["questions"] if q["type"] == "financial_premise"]
        agrees = [("늘었어" in q["question"]) == ("증가" in q["rubric"].split(".")[0]) for q in premise]
        self.assertTrue(any(agrees) and not all(agrees))


if __name__ == "__main__":
    unittest.main()
