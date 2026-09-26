"""캐시 모듈 단위 테스트 (임시 파일 사용, 네트워크 없음). 실행: python -m unittest discover tests"""

import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from src.data import cache


class TestCache(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._real = cache.CACHE_PATH
        cache.CACHE_PATH = Path(self._tmp.name) / "test_cache.db"  # 실제 캐시는 건드리지 않음

    def tearDown(self):
        cache.CACHE_PATH = self._real
        self._tmp.cleanup()

    def test_put_get_and_expiry(self):
        cache.put("ns", "k1", {"a": 1})  # 만료 없음
        cache.put("ns", "k2", [1, 2], ttl_seconds=0.05)
        self.assertEqual(cache.get("ns", "k1"), {"a": 1})
        self.assertEqual(cache.get("ns", "k2"), [1, 2])
        time.sleep(0.1)
        self.assertIsNone(cache.get("ns", "k2"))  # 만료된 값은 없는 것으로 본다
        self.assertIsNone(cache.get("other", "k1"))  # namespace가 다르면 다른 키

    def test_key_is_order_insensitive_for_dicts(self):
        self.assertEqual(cache.make_key({"a": 1, "b": 2}), cache.make_key({"b": 2, "a": 1}))
        self.assertNotEqual(cache.make_key("x", None), cache.make_key("x", "2026-01-01"))

    def test_embeddings_roundtrip(self):
        vec = np.random.rand(1024).astype(np.float32)
        cache.put_embeddings({"e1": vec})
        got = cache.get_embeddings(["e1", "missing"])
        self.assertEqual(set(got), {"e1"})
        np.testing.assert_allclose(got["e1"], vec)


if __name__ == "__main__":
    unittest.main()
