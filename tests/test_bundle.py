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
                "events": [{"id": 1, "t": 30, "type": "kill", "team": "red"}, {"id": 2, "t": 50.2, "type": "tower", "team": "blue"}],
                "signals": {"step_sec": 1.0, "start_sec": 0.0, "gold_diff": [None] * 40 + [500.0] * 61}})
            stage(work, "asr", {"lines": [{"id": 1, "start_sec": 25, "end_sec": 27, "text": "럼블", "raw": "런블", "fixes": [{}]}]})
            stage(work, "roi_ocr", {"runs": [{"tag": "team_kills", "slot": "red", "start_sec": 30, "end_sec": 60, "text": "1"}]})
            stage(work, "opgg_timeline", {"kills": [{"t": 10.0, "side": "red"}, {"t": 70.0, "side": "red"}],
                                           "objectives": [{"t": 35.0, "type": "herald", "side": "blue", "game_time": "00:35"}]})
            d = bundle.build(root)
        self.assertEqual([(p["fields"]["kind"], len(p["children"])) for p in d["segments"]],
                         [("draft", 1), ("game", 2), ("countdown", 1)])
        live, replay = d["segments"][1]["children"]
        self.assertEqual(([e["id"] for e in live["fields"]["events"]], [e["id"] for e in replay["fields"]["events"] if not e.get("filled")]),
                         ([1, 2], []))  # 전환 구간(50~50.5)의 타워는 앞 컷
        self.assertEqual(live["fields"]["lines"], [{"id": 1, "from_sec": 25, "to_sec": 27, "text": "럼블", "raw": "런블", "fixes": [{}]}])
        self.assertEqual(live["fields"]["screen"], {"kills": {"red": 1}, "gold_diff": 500.0})
        # op.gg 킬: 경기 10초 → 영상 30초 장면에 (우리 킬 이벤트 1번과 짝), 경기 70초는 clock 에서 20초 넘게 벗어나 밖으로
        ref = live["fields"]["reference"]
        self.assertEqual([(r["type"], r["video_sec"], r["matched_event"], r["diff_sec"]) for r in ref], [("kill", 30.0, 1, 0.0)])
        # 경기 35초 전령 = 영상 55초, 리플레이 장면이고 우리 이벤트가 없어 op.gg 로 채움
        filled = [e for e in replay["fields"]["events"] if e.get("filled")]
        self.assertEqual([(e["type"], e["subtype"], e["team"], e["source"]) for e in filled],
                         [("objective", "herald", "blue", ["opgg"])])
        self.assertNotIn("reference_outside", d)
        self.assertNotIn("ocr", live["fields"])
        self.assertEqual(live["fields"]["signals"]["gold_diff"][-1], 500.0)
        self.assertEqual(d["header"]["context"], {"match": {"league": "LCK"}})
        self.assertNotIn("run_log", d)
