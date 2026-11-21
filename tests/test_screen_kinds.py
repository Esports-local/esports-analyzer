"""segments 화면 종류 판별 (OCR 없이 규칙·다수결·경기 구간만)."""

import unittest

import numpy as np

from pipeline.screen_kinds import _contains, classify_frame, game_range, label_segments, load_rules

CONFIG = {"sample_sec": 2, "default": "break", "rules": [
    {"kind": "replay", "roi": [0, 0, 10, 10], "contains": "HIGHLIGHT"},
    {"kind": "live", "roi": [10, 0, 20, 10], "regex": r"^\d{1,2}:\d{2}$", "char_map": {"o": "0"}, "value": "game_time"},
    {"kind": "postgame", "roi": [20, 0, 30, 10], "contains": "GAMETIME",
     "read": {"game_length": {"roi": [0, 10, 10, 20], "regex": r"^\d{1,2}:\d{2}$"}}},
]}


class FakeReader:
    """자리(x 시작)마다 정해 둔 글자를 돌려준다."""

    def __init__(self, texts):
        self.texts = texts

    def __call__(self, image):
        return self.texts.get(int(image[0, 0, 0]), ""), 0.9


def frame():
    f = np.zeros((20, 30, 3), np.uint8)
    f[0:10, 0:10] = 1
    f[0:10, 10:20] = 2
    f[0:10, 20:30] = 3
    f[10:20, 0:10] = 4
    return f


class ScreenKindsTest(unittest.TestCase):
    def setUp(self):
        self.rules, self.default, self.step = load_rules(CONFIG)

    def test_classify_first_rule_wins(self):
        self.assertEqual(classify_frame(frame(), self.rules, self.default, FakeReader({2: "1o:05"})),
                         {"kind": "live", "game_time": "10:05"})
        self.assertEqual(classify_frame(frame(), self.rules, self.default, FakeReader({1: "HIGHL1GHT", 2: "10:05"})),
                         {"kind": "replay"})
        self.assertEqual(classify_frame(frame(), self.rules, self.default, FakeReader({3: "GAME TIME", 4: "38:45"})),
                         {"kind": "postgame", "game_length": "38:45"})
        self.assertEqual(classify_frame(frame(), self.rules, self.default, FakeReader({})), {"kind": "break"})

    def test_contains_tolerates_ocr_errors(self):
        self.assertTrue(_contains("GAMETlME", "GAMETIME"))
        self.assertFalse(_contains("GEN", "GAMETIME"))

    def test_label_segments_majority(self):
        samples = [{"t": 0, "kind": "break"}, {"t": 2, "kind": "live", "game_time": "00:01"},
                   {"t": 4, "kind": "live", "game_time": "00:03"}, {"t": 6, "kind": "break"}]
        segs = label_segments([{"start_sec": 0, "end_sec": 6}, {"start_sec": 6, "end_sec": 6.5}], samples, "break")
        self.assertEqual((segs[0]["kind"], segs[0]["game_start"], segs[0]["game_end"]), ("live", "00:01", "00:03"))
        self.assertEqual(segs[1]["kind"], "break")
        # 화면이 안 든 짧은 segment 는 가장 가까운 화면
        self.assertEqual(label_segments([{"start_sec": 4.2, "end_sec": 4.5}], samples, "break")[0]["kind"], "live")

    def test_contains_any_word(self):
        rules, default, _ = load_rules({"rules": [{"kind": "replay", "roi": [0, 0, 10, 10], "contains": ["REPLAY", "HIGHLIGHT"]}]})
        for text, kind in [("REPLAY", "replay"), ("HlGHLIGHT", "replay"), ("LIVE", "break")]:
            self.assertEqual(classify_frame(frame(), rules, default, FakeReader({1: text}))["kind"], kind)

    def test_game_range_with_header_and_postgame(self):
        samples = [{"t": 100, "kind": "live", "game_time": "00:10"}, {"t": 2400, "kind": "live", "game_time": "38:40"},
                   {"t": 2450, "kind": "postgame", "game_length": "38:45", "blue_result": "LOSS", "red_result": "WIN"}]
        g = game_range(samples, 2, {"result": {"length_sec": 2325, "winner": "red"}})
        self.assertEqual((g["start_sec"], g["end_sec"]), (90, 2405))
        self.assertEqual(g["postgame"], {"start_sec": 2450, "game_length_sec": 2325, "winner": "red"})
        self.assertEqual(g["checks"], {"length_matches_header": True, "winner_matches_header": True})
        self.assertIsNone(game_range([{"t": 0, "kind": "break"}], 2))
