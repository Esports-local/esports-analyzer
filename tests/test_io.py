"""단계 결과 공통 함수: 건너뛸 구간 (segments 화면 종류)."""

import json
import tempfile
import unittest
from pathlib import Path

from pipeline.io import in_spans, skip_spans


class SkipSpansTest(unittest.TestCase):
    def test_kinds_merged(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "work").mkdir()
            segs = [{"start_sec": 0, "end_sec": 10, "kind": "draft"}, {"start_sec": 10.2, "end_sec": 20, "kind": "break"},
                    {"start_sec": 20.5, "end_sec": 50, "kind": "live"}, {"start_sec": 50, "end_sec": 60, "kind": "countdown"}]
            (work / "work" / "segments.json").write_text(json.dumps({"result": {"segments": segs}}), encoding="utf-8")
            spans = skip_spans(work, ["draft", "break", "countdown"])
            self.assertEqual(spans, [(0, 20), (50, 60)])
            self.assertEqual((in_spans(spans, 15), in_spans(spans, 30)), (True, False))
            self.assertEqual(skip_spans(work, []), [])
            self.assertEqual(skip_spans(Path(tmp) / "없음", ["draft"]), [])
            # 경기 구간(20~50) 안의 break 는 건너뛰지 않고, in_game 에 준 replay 만 건너뜀
            segs2 = [{"start_sec": 0, "end_sec": 20, "kind": "draft"}, {"start_sec": 25, "end_sec": 30, "kind": "break"},
                     {"start_sec": 30, "end_sec": 35, "kind": "replay"}]
            (work / "work" / "segments.json").write_text(
                json.dumps({"result": {"segments": segs2, "game": {"start_sec": 20, "end_sec": 50}}}), encoding="utf-8")
            self.assertEqual(skip_spans(work, ["draft", "break", "replay"], ["replay"]), [(0, 20), (30, 35)])
