"""숏폼 렌더 규칙과 /shorts 줄 세우기 테스트. ffmpeg 는 돌리지 않는다 (render 를 바꿔 끼움)."""

import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from api.shorts import views
from app.main import app
from pipeline.shorts import _ass_events, _ass_time, _check_segments


class RenderRulesTest(unittest.TestCase):
    def test_subtitles_move_to_short_time(self):
        segs = [(100.0, 110.0), (200.0, 205.0)]
        subs = [
            {"start_sec": 98, "end_sec": 101, "text": "걸친 대사"},     # 앞이 잘림 → 0~1초
            {"start_sec": 150, "end_sec": 152, "text": "구간 밖"},       # 버림
            {"start_sec": 201, "end_sec": 203, "text": "둘째 구간"},     # 10 + 1 ~ 10 + 3
            {"start_sec": 109.9, "end_sec": 112, "text": "너무 짧음"},   # 구간 안이 0.1초 → 버림
        ]
        self.assertEqual(_ass_events(segs, subs), [(0.0, 1.0, "걸친 대사"), (11.0, 13.0, "둘째 구간")])

    def test_segments_checked(self):
        with self.assertRaises(ValueError):
            _check_segments([])
        with self.assertRaises(ValueError):
            _check_segments([{"start_sec": 10, "end_sec": 20}, {"start_sec": 15, "end_sec": 25}])  # 겹침
        with self.assertRaises(ValueError):
            _check_segments([{"start_sec": 0, "end_sec": 121}])  # 너무 김

    def test_ass_time(self):
        self.assertEqual(_ass_time(3725.456), "1:02:05.46")


class ShortsApiTest(unittest.TestCase):
    def setUp(self):
        views.SHORTS.clear()
        views.QUEUE.clear()
        self.client = TestClient(app)
        self.body = {"key": "abc12345", "video": "game.mp4",
                     "segments": [{"start_sec": 10, "end_sec": 20}], "subtitles": [], "title": "T1"}
        patcher = mock.patch.object(views, "video_path", side_effect=lambda name: Path("/data") / name)
        patcher.start()
        self.addCleanup(patcher.stop)
        disk = mock.patch.object(views, "_from_disk", return_value=None)
        disk.start()
        self.addCleanup(disk.stop)

    def _wait(self, key, status, timeout=3.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            state = self.client.get(f"/shorts/{key}").json()
            if state["status"] == status:
                return state
            time.sleep(0.02)
        self.fail(f"{key} 가 {status} 가 되지 않음: {state}")

    def test_render_then_same_request_returns_existing(self):
        with mock.patch.object(views, "render", return_value={"duration_sec": 10.0, "render_sec": 2.5, "size_bytes": 9}) as r:
            self.assertEqual(self.client.post("/shorts", json=self.body).status_code, 200)
            done = self._wait("abc12345", "succeeded")
            self.assertEqual((done["render_sec"], done["output"]), (2.5, "game/shorts/abc12345.mp4"))
            again = self.client.post("/shorts", json=self.body).json()
            self.assertEqual(again["status"], "succeeded")
            self.assertEqual(r.call_count, 1)  # 다시 렌더하지 않음

    def test_queue_waits_instead_of_409(self):
        gate = threading.Event()
        with mock.patch.object(views, "render", side_effect=lambda *a, **k: gate.wait(3) and {"duration_sec": 1}):
            self.client.post("/shorts", json=self.body)
            self._wait("abc12345", "rendering")
            second = self.client.post("/shorts", json={**self.body, "key": "def67890"}).json()
            self.assertEqual((second["status"], second["queue_position"]), ("queued", 0))
            gate.set()
            self._wait("def67890", "succeeded")

    def test_failed_is_retried_and_conflict_is_409(self):
        with mock.patch.object(views, "render", side_effect=RuntimeError("ffmpeg 실패")):
            self.client.post("/shorts", json=self.body)
            failed = self._wait("abc12345", "failed")
            self.assertIn("ffmpeg", failed["error"])
        with mock.patch.object(views, "render", return_value={"duration_sec": 10.0}):
            retry = self.client.post("/shorts", json=self.body).json()
            self.assertEqual(retry["status"], "queued")
            self.assertEqual(self._wait("abc12345", "succeeded")["attempts"], 2)
        other = {**self.body, "title": "다른 제목"}
        self.assertEqual(self.client.post("/shorts", json=other).status_code, 409)

    def test_bad_segments_400(self):
        body = {**self.body, "segments": [{"start_sec": 10, "end_sec": 20}, {"start_sec": 15, "end_sec": 30}]}
        self.assertEqual(self.client.post("/shorts", json=body).status_code, 400)


if __name__ == "__main__":
    unittest.main()
