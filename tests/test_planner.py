"""agent/planner.py 단위 테스트 (OpenAI 클라이언트를 가짜로 바꿔 API 호출 없이 검증).

이전까지 planner.py(에이전트 루프의 핵심 파일)는 테스트가 하나도 없었다. 실제 LLM 응답을
흉내 낸 가짜 클라이언트로, 도구 호출 총량 상한(MAX_TOOL_CALLS)이 실제로 루프를 끊는지와
검증 단계(verify) on/off 스위치가 의도대로 동작하는지만 확인한다 (수치 정확도 등은 채점기
테스트의 몫).

실행: python -m unittest discover tests
"""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.agent import planner


class FakeToolCall:
    def __init__(self, call_id: str, name: str, args: dict):
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=json.dumps(args))

    def model_dump(self) -> dict:
        return {"id": self.id, "type": "function", "function": {"name": self.function.name, "arguments": self.function.arguments}}


class FakeClient:
    """tools=... 가 있으면 매번 도구 호출 4개를 요청하고, 없으면 고정 문구로 답한다."""

    def __init__(self, n_parallel_calls: int = 4, final_text: str = "최종 답변(모의)"):
        self.n_parallel_calls = n_parallel_calls
        self.final_text = final_text
        self.calls_with_tools = 0
        self.calls_without_tools = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, messages, tools=None, **kwargs):
        if tools:
            self.calls_with_tools += 1
            calls = [
                FakeToolCall(f"call{self.calls_with_tools}_{i}", "dummy_tool", {"i": i})
                for i in range(self.n_parallel_calls)
            ]
            message = SimpleNamespace(content=None, tool_calls=calls)
        else:
            self.calls_without_tools += 1
            message = SimpleNamespace(content=self.final_text, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


class TestToolCallBudget(unittest.TestCase):
    """병렬 호출이 몰리면 스텝 상한(7)보다 먼저 총 호출 수 상한(MAX_TOOL_CALLS)에 걸려야 한다."""

    def setUp(self):
        self.fake_client = FakeClient(n_parallel_calls=4)
        patcher_client = patch.object(planner, "_get_client", return_value=self.fake_client)
        patcher_tools = patch.dict(planner.TOOL_FUNCTIONS, {"dummy_tool": lambda i: "ok"})
        self.addCleanup(patcher_client.stop)
        self.addCleanup(patcher_tools.stop)
        patcher_client.start()
        patcher_tools.start()

    def test_stops_at_tool_call_budget_not_step_budget(self):
        result = planner.run_agent("아무 질문")
        # 4개씩 5스텝 = 20개에서 상한(20)에 도달 - 스텝 상한(7)까지 가지 않고 먼저 끊긴다.
        self.assertEqual(len(result["transcript"]), planner.MAX_TOOL_CALLS)
        self.assertEqual(result["n_steps"], planner.MAX_TOOL_CALLS // 4)
        self.assertLessEqual(result["n_steps"], planner.MAX_ITERATIONS)

    def test_verify_false_skips_verification_pass(self):
        result = planner.run_agent("아무 질문", verify=False)
        self.assertEqual(result["answer"], result["draft_answer"])

    def test_verify_true_still_returns_an_answer(self):
        result = planner.run_agent("아무 질문", verify=True)
        self.assertTrue(result["answer"])


class TestNeedsFinancialData(unittest.TestCase):
    def test_financial_question_without_tool_call_needs_guard(self):
        self.assertTrue(planner._needs_financial_data("영업이익이 왜 늘었어?", []))

    def test_financial_question_after_tool_call_does_not_need_guard(self):
        transcript = [{"tool": "get_financial_data", "args": {}, "result": "..."}]
        self.assertFalse(planner._needs_financial_data("영업이익이 왜 늘었어?", transcript))

    def test_non_financial_question_never_needs_guard(self):
        self.assertFalse(planner._needs_financial_data("오늘 날씨 어때?", []))


if __name__ == "__main__":
    unittest.main()
