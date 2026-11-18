"""HTTP 서버 스모크 테스트. 분석 단계는 돌리지 않는다 (CI 에 torch 등을 설치하지 않음)."""

import unittest

from fastapi.testclient import TestClient

from app.main import app
from pipeline.stages import IMPLEMENTED, STAGES


class AppTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["stages"], list(STAGES))
        self.assertEqual(res.json()["implemented"], list(IMPLEMENTED))

    def test_openapi(self):
        paths = self.client.get("/openapi.json").json()["paths"]
        for path in ("/videos", "/jobs", "/estimate"):
            self.assertTrue(any(p.startswith(path) for p in paths), path)


if __name__ == "__main__":
    unittest.main()
