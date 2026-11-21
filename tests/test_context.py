"""context(바디): 시계 · 이벤트 태그 · 신호 (영상·OCR 없이 결과 JSON 을 만들어 넣는다)."""

import unittest

from pipeline.game.context import (DEFAULT_PARAMS, build_clock, build_context, build_signals, game_to_video,
                                   video_to_game)


def gt(start, end, text):
    return {"tag": "game_time", "slot": "", "start_sec": start, "end_sec": end, "text": text}


def kill(i, t, team, killer):
    return {"id": i, "start_sec": t, "end_sec": t, "type": "kill", "team": team, "game_time": None,
            "players": {"killer": [killer], "victims": [], "assists": []}, "line_ids": [], "evidence": []}


class ContextTest(unittest.TestCase):
    def test_clock_splits_on_replay(self):
        # 100초에 경기 00:10, 200~215 리플레이(안 읽힘) → 끊김, 215초 경기 02:05 (차이 그대로)
        # 250초 1초짜리 오독은 버리고 앞뒤를 이음. 305초 경기 00:24 는 거꾸로 가서 버림. 300~320초 경기 멈춤(차이 +20) → 새 구간
        runs = [gt(100, 101, "00:10"), gt(101, 200, "00:11"), gt(215, 250, "02:05"), gt(250, 251, "09:59"),
                gt(251, 300, "02:41"), gt(305, 315, "00:24"), gt(320, 400, "03:30")]
        clock = build_clock(runs, 1.5, 3.0)
        self.assertEqual(clock, [{"start_sec": 100, "end_sec": 200, "game_start_sec": 10},
                                 {"start_sec": 215, "end_sec": 300, "game_start_sec": 125},
                                 {"start_sec": 320, "end_sec": 400, "game_start_sec": 210}])
        self.assertEqual((video_to_game(clock, 150), video_to_game(clock, 205), video_to_game(clock, 310),
                          game_to_video(clock, 220)), (60, None, None, 330))

    def test_events_tags_and_slots(self):
        tl = {"players": {"blue_top": "Kin", "red_jungle": "Kanavi"},
              "events": [kill(1, 100, "red", "Kanavi"), kill(2, 105, "red", "Kanavi"), kill(3, 112, "blue", "Kin"),
                         kill(4, 300, "blue", "Kin")]}
        header = {"players": [{"side": "red", "role": "jungle", "team": "HLE", "name": "Kanavi",
                               "champion": {"name": "Naafiri", "ko": "나피리"}}], "result": {"winner": "red", "length_sec": 2325}}
        segments = {"transitions": [{"id": 1, "kind": "wipe", "start_sec": 99, "end_sec": 99.5}],
                    "segments": [{"id": 1, "start_sec": 0, "end_sec": 99, "kind": "draft"},
                                 {"id": 2, "start_sec": 99.5, "end_sec": 400, "kind": "live", "transition_in": 1,
                                  "game_start": "00:00", "game_end": "05:00"}],
                    "game": {"start_sec": 99.5, "end_sec": 400}}
        c = build_context(header, [], tl, segments, [], 400, None, DEFAULT_PARAMS)
        tags = {e["id"]: e["tags"] for e in c.events}
        self.assertEqual(tags, {1: ["first_blood", "teamfight:1"], 2: ["multikill:2", "teamfight:1"],
                                3: ["teamfight:1"], 4: []})
        self.assertEqual((c.events[0]["killer"], c.events[0]["segment"]), ("red_jungle", "live"))
        self.assertEqual(c.players["red_jungle"], {"team": "HLE", "name": "Kanavi", "champion": "나피리", "champion_en": "Naafiri"})
        self.assertEqual(c.segments[1]["transition_in"], "wipe")
        self.assertEqual((c.game["winner"], c.game["length_sec"], c.game["start_sec"]), ("red", 2325, 99.5))

    def test_signals(self):
        runs = [{"tag": "gold", "slot": "blue", "start_sec": 1, "end_sec": 3, "text": "5.0K"},
                {"tag": "gold", "slot": "red", "start_sec": 2, "end_sec": 4, "text": "4.5K"}]
        segs = [{"start_sec": 0, "end_sec": 5, "kind": "live"}]
        s = build_signals(5, runs, {"transitions": [{"start_sec": 2.0}]}, segs, [0.1] * 10, DEFAULT_PARAMS)
        self.assertEqual(s["gold_diff"], [None, None, 500.0, 500.0, 500.0, 500.0])
        self.assertEqual((s["cut_rate"], s["kind"][0], len(s["loudness"])), ([1] * 6, "live", 6))

    def test_signals_start_at_range(self):
        runs = [{"tag": "gold", "slot": "blue", "start_sec": 901, "end_sec": 903, "text": "5.0K"},
                {"tag": "gold", "slot": "red", "start_sec": 901, "end_sec": 903, "text": "4.0K"}]
        s = build_signals(904, runs, {"transitions": [{"start_sec": 902.0}]}, [], [0.5] * 5, DEFAULT_PARAMS, start=900)
        self.assertEqual((s["start_sec"], len(s["gold_diff"]), s["gold_diff"][:2], len(s["loudness"])),
                         (900, 5, [None, 1000.0], 5))
        self.assertEqual(s["cut_rate"], [1] * 5)
