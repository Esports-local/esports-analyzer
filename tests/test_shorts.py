"""숏폼 렌더 규칙과 /shorts 줄 세우기 테스트. ffmpeg 는 돌리지 않는다 (render 를 바꿔 끼움)."""

import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from api.shorts import views
from app.main import app
from pipeline.shorts.render import _check_segments


class RenderRulesTest(unittest.TestCase):
    def test_segments_checked(self):
        with self.assertRaises(ValueError):
            _check_segments([])
        with self.assertRaises(ValueError):
            _check_segments([{"start_sec": 10, "end_sec": 20}, {"start_sec": 15, "end_sec": 25}])  # 겹침
        self.assertEqual(len(_check_segments([{"start_sec": i * 10, "end_sec": i * 10 + 9} for i in range(20)])), 20)  # 제한 없음


class ShortsApiTest(unittest.TestCase):
    def setUp(self):
        views.SHORTS.clear()
        views.QUEUE.clear()
        self.client = TestClient(app)
        self.body = {"key": "abc12345", "video": "game.mp4",
                     "segments": [{"start_sec": 10, "end_sec": 20}], "title": "T1"}
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


class RenderLayoutTest(unittest.TestCase):
    def test_caption_times_follow_joined_segments(self):
        from pipeline.shorts.render import output_spans
        segs = [(100.0, 110.0), (200.0, 210.0)]
        self.assertEqual(output_spans(segs, 105, 108), [(5.0, 8.0)])
        self.assertEqual(output_spans(segs, 108, 205), [(8.0, 10.0), (10.0, 15.0)])  # 두 구간에 걸침
        self.assertEqual(output_spans(segs, 150, 160), [])  # 구간 밖

    def test_command_has_layout_and_text(self):
        import tempfile
        from pipeline.shorts.render import _ffmpeg_cmd, _text_layers
        segs = [(100.0, 110.0)]
        with tempfile.TemporaryDirectory() as tmp:
            texts = _text_layers(segs, ["2026 LCK 결승 1세트", "듀로의 즉시 복수"],
                                 [{"start_sec": 104, "end_sec": 108, "lines": ["Kanavi, Ruler 잡고", "킬 GEN 3 : 4 HLE"]}],
                                 Path(tmp))
            cmd = _ffmpeg_cmd(Path("v.mp4"), segs, Path("o.mp4"), 23, "veryfast", texts)
            graph = cmd[cmd.index("-filter_complex") + 1]
        self.assertIn("crop=1180:1080:370:0", graph)
        self.assertIn("overlay=0:320", graph)
        self.assertEqual(graph.count("drawtext"), 4)
        self.assertIn("enable='between(t,4.0,8.0)'", graph)


class BgmTest(unittest.TestCase):
    def test_pick_is_stable_and_listing_wins(self):
        import json as _json
        import tempfile
        from pipeline.shorts import bgm
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertIsNone(bgm.pick("sabc", root))  # 곡 없음 → BGM 없이
            for n in ("a.mp3", "b.mp3", "c.txt"):
                (root / n).write_bytes(b"x")
            self.assertEqual([Path(t["file"]).name for t in bgm.tracks(root)], ["a.mp3", "b.mp3"])
            self.assertEqual(bgm.pick("sabc", root), bgm.pick("sabc", root))
            (root / "bgm.json").write_text(_json.dumps([{"file": "b.mp3", "title": "B", "attribution": "B by X"},
                                                        {"file": "없는.mp3", "title": "?"}]), encoding="utf-8")
            self.assertEqual([t["title"] for t in bgm.tracks(root)], ["B"])  # 목록에 있고 파일이 있는 것만

    def test_command_mixes_bgm_under_voice(self):
        from pipeline.shorts.render import _ffmpeg_cmd
        cmd = _ffmpeg_cmd(Path("v.mp4"), [(0.0, 10.0)], Path("o.mp4"), 23, "veryfast", [], "/b/song.mp3")
        graph = cmd[cmd.index("-filter_complex") + 1]
        self.assertIn("-stream_loop", cmd)
        self.assertIn("sidechaincompress", graph)
        self.assertIn("[aout]", cmd)
        plain = _ffmpeg_cmd(Path("v.mp4"), [(0.0, 10.0)], Path("o.mp4"), 23, "veryfast", [])
        self.assertIn("[ca]", plain)
