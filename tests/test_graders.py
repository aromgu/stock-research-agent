"""채점기 단위 테스트 (API·DB 호출 없음). 실행: python -m unittest discover tests

테스트 사례는 실제 평가 중 문제가 됐던 답변 표현에서 가져왔다.
"""

import unittest

from src.eval.graders import (
    _stated_direction_matches,
    accumulate_citation_counts,
    describe_numeric_check,
    extract_candidate_numbers,
    extract_multiples,
    extract_percentages,
    is_deferred_answer,
    precision_recall_f1,
)
from src.eval.trajectory import analyze_trajectory


class TestNumberExtraction(unittest.TestCase):
    def test_korean_units_with_duplicated_unit_typo(self):
        # 실제 에이전트 답변: 금액은 맞는데 "만원 원" 오타가 있었음 (LLM 심판이 오판했던 사례)
        self.assertIn(305_372_914_000_000, extract_candidate_numbers("누적 매출액은 305조 3,729억 1,400만원 원입니다"))

    def test_comma_before_korean_word_is_not_a_number(self):
        # 실제로 채점을 멈추게 한 버그: 쉼표 뒤 '억제' 같은 단어를 ", 억"이라는 금액으로 읽으려다 ValueError
        self.assertEqual(extract_candidate_numbers("원가 상승, 억제 효과와 비용, 만큼의 여유"), [])
        self.assertIn(1_500_000_000_000, extract_candidate_numbers("비용 절감, 1.5조원 수준"))

    def test_plain_comma_won(self):
        self.assertIn(98_152_891_000_000, extract_candidate_numbers("영업이익은 98,152,891,000,000원입니다"))

    def test_decimal_jo(self):
        self.assertIn(146_700_000_000_000, extract_candidate_numbers("약 146.7조원"))

    def test_percentages_with_sign(self):
        self.assertEqual(extract_percentages("수익률 -4.21%, 영업이익률 48.1%, 증가율 98.67퍼센트"), [-4.21, 48.1, 98.67])

    def test_multiples(self):
        self.assertEqual(extract_multiples("PER 43.22배, PBR 4.46배"), [43.22, 4.46])


class TestGrowthAcceptsAmounts(unittest.TestCase):
    def test_growth_by_amounts_or_percent(self):
        from unittest.mock import patch

        from src.eval import graders

        nt = {"kind": "growth", "company": "X", "bsns_year": "2026", "reprt_code": "11012", "account_name": "영업이익"}
        with patch.object(graders, "_growth_amounts", return_value=(98_152_891_000_000, 16_653_355_000_000)):
            by_amounts = "전년 동기 16조 6,533억원 → 당기 98조 1,529억원으로 증가"
            by_percent = "전년 대비 약 489% 증가"
            wrong = "전년 115조 8,336억원에서 98조 1,529억원으로 감소"
            self.assertTrue(graders.grade_numeric(by_amounts, nt)["correct"])
            self.assertTrue(graders.grade_numeric(by_percent, nt)["correct"])
            self.assertFalse(graders.grade_numeric(wrong, nt)["correct"])


class TestDirection(unittest.TestCase):
    def test_correction_mentions_true_direction(self):
        # 잘못된 전제("왜 올랐어?")를 바로잡는 답변은 질문을 되풀이하며 "올랐"도 포함한다 — 참 방향 단어가 있으면 정답
        answer = "먼저 전제가 틀렸습니다. 올랐던 게 아니라 오히려 하락했습니다(-4.21%)."
        self.assertTrue(_stated_direction_matches(answer, "down"))

    def test_accepting_false_premise(self):
        self.assertFalse(_stated_direction_matches("반도체 호황으로 주가가 올랐습니다.", "down"))


class TestDeferredAnswer(unittest.TestCase):
    def test_ends_with_clarifying_question(self):
        answer = "영업이익은 146.7조원입니다. 원인은 기사를 못 찾았습니다.\n어떤 관점이 더 필요할까요?"
        self.assertTrue(is_deferred_answer(answer))

    def test_normal_answer(self):
        self.assertFalse(is_deferred_answer("삼성전자 매출액은 305조원입니다."))


class TestCitation(unittest.TestCase):
    def test_counts_and_f1(self):
        counts = {"tp": 0, "fp": 0, "fn": 0}
        accumulate_citation_counts({"dart", "news", "price"}, {"dart"}, counts)  # 항상 3도구 호출
        accumulate_citation_counts({"dart"}, {"dart", "news"}, counts)  # 뉴스를 빠뜨림
        self.assertEqual(counts, {"tp": 2, "fp": 2, "fn": 1})
        pr = precision_recall_f1(counts)
        self.assertAlmostEqual(pr["precision"], 0.5)
        self.assertAlmostEqual(pr["recall"], 2 / 3)

    def test_numeric_check_message(self):
        self.assertIsNone(describe_numeric_check({"applicable": False, "correct": None}))
        self.assertIn("일치하는 수치가 있음", describe_numeric_check({"applicable": True, "correct": True}))


class TestFinancialGuard(unittest.TestCase):
    def test_guard_condition(self):
        from src.agent.planner import _needs_financial_data

        news_only = [{"tool": "search_news"}]
        with_dart = [{"tool": "search_news"}, {"tool": "get_financial_data"}]
        self.assertTrue(_needs_financial_data("상반기 영업이익이 왜 줄었어?", news_only))
        self.assertFalse(_needs_financial_data("상반기 영업이익이 왜 줄었어?", with_dart))
        self.assertFalse(_needs_financial_data("주가가 왜 올랐어?", news_only))  # 재무 질문이 아니면 강제 안 함


class TestTrajectory(unittest.TestCase):
    def test_agent_duplicates_and_empty_results(self):
        run = {
            "predicted_sources": ["news"],
            "context": "",
            "transcript": [
                {"step": 1, "tool": "search_news", "args": {"query": "a"}, "result": "- [1] 기사"},
                {"step": 2, "tool": "search_news", "args": {"query": "a"}, "result": "- [1] 기사"},
                {"step": 3, "tool": "search_news", "args": {"query": "b"}, "result": "관련 뉴스를 찾지 못했습니다."},
            ],
        }
        t = analyze_trajectory(run)
        self.assertEqual((t["tool_calls"], t["duplicate_calls"], t["empty_calls"], t["llm_steps"]), (3, 1, 1, 4))

    def test_fixed_pipeline(self):
        run = {"predicted_sources": ["dart", "news", "price"], "context": "[DART]\n값", "transcript": None}
        self.assertEqual(analyze_trajectory(run)["tool_calls"], 3)


if __name__ == "__main__":
    unittest.main()
