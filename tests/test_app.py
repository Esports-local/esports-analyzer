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
        for path in ("/videos", "/jobs", "/shorts"):
            self.assertTrue(any(p.startswith(path) for p in paths), path)


    def test_video_file(self):
        import tempfile
        from pathlib import Path
        from unittest import mock

        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            (root / "a.mp4").write_bytes(b"0123456789")
            (root / "notes.txt").write_text("x")
            with mock.patch("api.paths.VIDEOS_ROOT", root):
                res = self.client.get("/videos/a.mp4/file", headers={"Range": "bytes=2-4"})
                self.assertEqual(res.status_code, 206)
                self.assertEqual(res.content, b"234")
                self.assertEqual(res.headers["access-control-allow-origin"], "http://localhost:8000")
                self.assertEqual(self.client.get("/videos/a.mp4/file").content, b"0123456789")
                self.assertEqual(self.client.get("/videos/b.mp4/file").status_code, 404)
                self.assertEqual(self.client.get("/videos/notes.txt/file").status_code, 404)
                self.assertEqual(self.client.get("/videos/..%2Fa.mp4/file").status_code, 404)


if __name__ == "__main__":
    unittest.main()
