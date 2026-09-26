"""대화 메모리·기사 본문 추출 단위 테스트 (임시 DB, 네트워크·LLM 호출 없음)."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bs4 import BeautifulSoup

from src.data import article_client as ac
from src.data import cache


class _TempCache(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._real = cache.CACHE_PATH
        cache.CACHE_PATH = Path(self._tmp.name) / "test_cache.db"

    def tearDown(self):
        cache.CACHE_PATH = self._real
        self._tmp.cleanup()


class TestArticleExtraction(unittest.TestCase):
    BODY = "삼성전자는 2분기 영업이익이 크게 늘었다고 밝혔다. " * 10

    def test_naver_page(self):
        html = f"<html><body><div id='dic_area'>{self.BODY}</div><footer>저작권</footer></body></html>"
        self.assertIn("영업이익", ac._extract_naver(BeautifulSoup(html, "lxml")))

    def test_br_separated_body_beats_footer_paragraphs(self):
        # 본문을 <p> 없이 <br>로 쓰는 사이트에서 예전 규칙은 <p>로 된 하단 정보(사업자번호)를 본문으로 잡았다
        footer = "".join(f"<p>사업자번호 123-45-{i:05d} 등록번호 서울 아{i:05d}</p>" for i in range(8))
        html = f"<html><body><div class='body'>{self.BODY}<br>{self.BODY}</div><div class='foot'>{footer}</div></body></html>"
        text = ac._extract_generic(BeautifulSoup(html, "lxml"))
        self.assertIn("영업이익", text)
        self.assertNotIn("사업자번호", text)


class TestReadArticles(_TempCache):
    def test_unknown_id_and_cached_text(self):
        item = {"link": "https://n.news.naver.com/x", "title": "제목"}
        ac.remember_links([item])
        aid = ac.article_id(item["link"])
        with patch.object(ac, "fetch_article_text", return_value="본문 " * 1000):
            known, unknown = ac.read_articles([aid, "deadbeef"])
        self.assertEqual(known["title"], "제목")
        self.assertTrue(known["text"].endswith("…"))  # 긴 본문은 잘라서 넘김 (토큰 절약)
        self.assertIsNone(unknown["text"])


class TestConversationMemory(_TempCache):
    def test_first_turn_skips_llm_and_history_is_saved(self):
        from src.agent import memory

        self.assertEqual(memory.rewrite_question("삼성전자 영업이익은?", [], {}), "삼성전자 영업이익은?")
        fake = {"answer": "98조원입니다.", "transcript": [], "usage": {}}
        with patch.object(memory, "run_agent", return_value=fake):
            memory.ask("s1", "삼성전자 상반기 영업이익은?")
        with patch.object(memory, "run_agent", return_value=fake), patch.object(
            memory, "rewrite_question", return_value="SK하이닉스 상반기 영업이익은?"
        ) as rewrite:
            result = memory.ask("s1", "그럼 SK하이닉스는?")
        history_seen = rewrite.call_args[0][1]
        self.assertEqual([h["question"] for h in history_seen], ["삼성전자 상반기 영업이익은?"])
        self.assertEqual(result["standalone_question"], "SK하이닉스 상반기 영업이익은?")
        self.assertEqual(len(memory.load_history("s1")), 2)
        self.assertEqual(memory.load_history("other"), [])  # 세션끼리 섞이지 않음


if __name__ == "__main__":
    unittest.main()
