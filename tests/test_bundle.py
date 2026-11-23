"""analyzer.json 시간순 트리 (pipeline/game/bundle.py)."""

import json
import tempfile
import unittest
from pathlib import Path

from pipeline.game import bundle


def stage(work, name, result):
    (work / f"{name}.json").write_text(json.dumps({"stage": name, "result": result}, ensure_ascii=False), encoding="utf-8")


class BundleTest(unittest.TestCase):
    def test_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "v"
            work = root / "work"
            work.mkdir(parents=True)
            stage(work, "preprocess", {"duration_sec": 100, "range_start_sec": 0})
            stage(work, "header", {"match": {"league": "LCK"}})
            stage(work, "context", {
                "game": {"start_sec": 20, "end_sec": 90},
                "clock": [{"start_sec": 20, "end_sec": 60, "game_start_sec": 0}],
                "segments": [{"id": 1, "start_sec": 0, "end_sec": 20, "kind": "draft"},
                             {"id": 2, "start_sec": 20, "end_sec": 50, "kind": "live", "transition_in": "cut", "game_time": ["00:00", "00:30"]},
                             {"id": 3, "start_sec": 50.5, "end_sec": 70, "kind": "replay", "transition_in": "wipe"},
                             {"id": 4, "start_sec": 95, "end_sec": 100, "kind": "countdown"}],
                "events": [{"id": 1, "t": 30, "type": "kill", "team": "red", "source": ["opgg kill 00:10"],
                            "position": {"x": 7400, "y": 7400}},
                           {"id": 2, "t": 50.2, "type": "tower", "team": "blue"}],
                "signals": {"step_sec": 1.0, "start_sec": 0.0, "gold_diff": [None] * 40 + [500.0] * 61}})
            stage(work, "asr", {"lines": [{"id": 1, "start_sec": 25, "end_sec": 27, "text": "럼블", "raw": "런블", "fixes": [{}]}]})
            stage(work, "timeline", {"source": "opgg", "sync": {"checked": 1, "matched": 1, "lag_median_sec": 1.5}})
            stage(work, "opgg_timeline", {"kills": [{"t": 10.0, "side": "red"}, {"t": 70.0, "side": "red"}],
                                           "gold": {"blue": [[0.0, 2500], [60.0, 3700]], "red": [[0.0, 2500], [60.0, 3100]]}})
            d = bundle.build(root)
        self.assertEqual([(p["fields"]["kind"], len(p["children"])) for p in d["segments"]],
                         [("draft", 1), ("game", 2), ("countdown", 1)])
        live, replay = d["segments"][1]["children"]
        self.assertEqual([e["id"] for e in live["fields"]["events"]], [1, 2])  # 전환 구간(50~50.5)의 타워는 앞 컷
        self.assertEqual(live["fields"]["events"][0]["area"], "mid_lane")
        self.assertEqual(live["fields"]["lines"], [{"id": 1, "from_sec": 25, "to_sec": 27, "text": "럼블", "raw": "런블", "fixes": [{}]}])
        # 컷 끝 50초 = 경기 30초: op.gg 킬 누적(경기 10초 1개), 골드는 화면 값 먼저
        self.assertEqual(live["fields"]["state"], {"kills": {"blue": 0, "red": 1}, "gold_diff": 500.0, "gold_source": "screen"})
        # 첫 컷 끝 20초: 화면 골드 없음 → op.gg 보간. 경기 0초 전이라 op.gg 범위 밖 → None
        self.assertEqual(d["segments"][0]["children"][0]["fields"]["state"]["gold_source"], "opgg")
        self.assertNotIn("reference", live["fields"])
        self.assertEqual((d["header"]["event_source"], d["header"]["sync"]["matched"]), ("opgg", 1))
        self.assertNotIn("reference_outside", d)
        self.assertNotIn("ocr", live["fields"])
        self.assertEqual(live["fields"]["signals"]["gold_diff"][-1], 500.0)
        self.assertEqual(d["header"]["context"], {"match": {"league": "LCK"}})
        self.assertNotIn("run_log", d)


class GoldTest(unittest.TestCase):
    def test_opgg_gold_interpolated_when_no_screen(self):
        ref = {"gold": {"blue": [[0.0, 2500], [60.0, 3700]], "red": [[0.0, 2500], [60.0, 3100]]}}
        clock = [{"start_sec": 20, "end_sec": 60, "game_start_sec": 0}]
        st = bundle._state([], ref, clock, {"gold_diff": [], "start_sec": 0.0, "step_sec": 1.0}, 70.0)  # 리플레이: 경기 50초
        self.assertEqual((st["gold_diff"], st["gold_source"]), (500, "opgg"))
