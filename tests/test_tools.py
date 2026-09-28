"""agent/tools.py의 순수 로직(문자열 포맷팅·업종명 매칭) 단위 테스트 (API·DB 호출 없음).

실행: python -m unittest discover tests
"""

import unittest

from src.agent import tools


class TestMatchSegments(unittest.TestCase):
    def test_exact_segment_name_only_matches_itself(self):
        # v3_3 버그: "장비"만으로 매칭하면 "전공정 장비"가 "후공정 장비"까지 잡았다.
        # 업종 이름을 통째로 말했으면 그 업종만 반환해야 한다.
        self.assertEqual(tools._match_segments("전공정 장비"), ["전공정 장비"])
        self.assertEqual(tools._match_segments("후공정 장비"), ["후공정 장비"])

    def test_partial_expression_matches_semiconductor_material_segment(self):
        # v3_3 버그: "반도체 소재"로 물으면 "소재"와 정확히 같지 않아 못 찾았다.
        self.assertEqual(tools._match_segments("반도체 소재 업체들"), ["소재"])

    def test_generic_word_without_segment_hint_matches_nothing(self):
        self.assertEqual(tools._match_segments("반도체 회사 중"), [])

    def test_suffix_ju_is_stripped_before_matching(self):
        self.assertIn("소재", tools._match_segments("소재주"))


class TestDescribeRow(unittest.TestCase):
    def _row(self, **overrides):
        base = {
            "account_name": "영업이익",
            "sj_name": "손익계산서",
            "thstrm_amount": 1000,
            "thstrm_add_amount": None,
            "frmtrm_amount": 800,
            "frmtrm_add_amount": None,
            "reprt_code": "11013",
        }
        base.update(overrides)
        return base

    def test_balance_sheet_uses_period_end_wording(self):
        row = self._row(sj_name="재무상태표", account_name="자산총계")
        line = tools._describe_row(row)
        self.assertIn("기말 시점 기준", line)
        self.assertIn("전기말", line)

    def test_annual_report_uses_annual_wording(self):
        row = self._row(reprt_code="11011")
        line = tools._describe_row(row)
        self.assertIn("연간 누적", line)
        self.assertIn("전년 연간", line)

    def test_quarterly_report_shows_both_standalone_and_cumulative(self):
        row = self._row(reprt_code="11012", thstrm_add_amount=5000, frmtrm_add_amount=4000)
        line = tools._describe_row(row)
        self.assertIn("해당 분기 단독", line)
        self.assertIn("연초~해당 분기 누적", line)
        self.assertIn("전년 동기 누적", line)


class TestSmallHelpers(unittest.TestCase):
    def test_won_formats_thousands_separator(self):
        self.assertEqual(tools._won(1234567), "1,234,567원")

    def test_won_none_is_none_label(self):
        self.assertEqual(tools._won(None), "없음")

    def test_month_end_day(self):
        self.assertEqual(tools._month_end_day(6), 30)
        self.assertEqual(tools._month_end_day(12), 31)


class TestCumulative(unittest.TestCase):
    def test_prefers_cumulative_field_over_standalone(self):
        row = {"thstrm_add_amount": 500, "thstrm_amount": 100, "frmtrm_add_amount": 400, "frmtrm_amount": 90}
        self.assertEqual(tools._cumulative(row), 500)
        self.assertEqual(tools._cumulative(row, prior=True), 400)

    def test_falls_back_to_standalone_when_no_cumulative_field(self):
        # 1분기·연간 보고서는 누적 필드가 없다(None) - 단독 값으로 대체
        row = {"thstrm_add_amount": None, "thstrm_amount": 100, "frmtrm_add_amount": None, "frmtrm_amount": 90}
        self.assertEqual(tools._cumulative(row), 100)
        self.assertEqual(tools._cumulative(row, prior=True), 90)


if __name__ == "__main__":
    unittest.main()
