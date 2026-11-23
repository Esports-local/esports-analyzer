"""timeline 단계 규칙 테스트 (roi_ocr 구간을 직접 만들어 넣는다. 영상·모델 불필요)."""

import unittest

from pipeline.game.timeline import DEFAULT_PARAMS, build_timeline, counter_steps


def run(tag, slot, start, end, text, samples=4):
    return {"tag": tag, "slot": slot, "start_sec": start, "end_sec": end, "text": text,
            "raw": text, "conf": 0.9, "samples": samples, "kind": None}


class CounterStepsTest(unittest.TestCase):
    def test_increase_becomes_step(self):
        series = [run("team_kills", "red", 0, 10, "0"), run("team_kills", "red", 10, 20, "1")]
        self.assertEqual([s[:4] for s in counter_steps(series, 2, 3)], [(0, 1, 10, 10)])

    def test_noise_drop_and_jump_ignored(self):
        series = [
            run("team_kills", "red", 0, 10, "2"),
            run("team_kills", "red", 10, 10.5, "3", samples=1),  # 1장짜리 → 잡음
            run("team_kills", "red", 11, 12, "1"),               # 줄어듦 → 오독
            run("team_kills", "red", 12, 13, "9"),               # 한 번에 +7 → 오독
            run("team_kills", "red", 13, 20, "3"),
        ]
        self.assertEqual([s[:4] for s in counter_steps(series, 2, 3)], [(2, 3, 10, 13)])

    def test_gap_keeps_range(self):
        # 리플레이로 30~60초를 못 읽었으면 그 사이 어딘가에서 일어난 것
        series = [run("towers", "blue", 0, 30, "0"), run("towers", "blue", 60, 90, "1")]
        self.assertEqual([s[2:4] for s in counter_steps(series, 2, 3)], [(30, 60)])


class BuildTimelineTest(unittest.TestCase):
    def test_kill_players_and_lines(self):
        runs = [
            run("team", "blue", 0, 100, "T1"), run("team", "red", 0, 100, "BLG"),
            run("player_name", "blue_top", 0, 100, "Doran"), run("player_name", "red_top", 0, 100, "Bin"),
            run("team_kills", "blue", 0, 50, "0"), run("team_kills", "blue", 50, 100, "1"),
            run("kda", "blue_top", 0, 50, "0/0/0"), run("kda", "blue_top", 51, 100, "1/0/0"),
            run("kda", "red_top", 0, 49, "0/0/0"), run("kda", "red_top", 49, 100, "0/1/0"),
            run("game_time", "", 49, 51, "03:24"),
        ]
        lines = [{"id": 1, "start_sec": 46, "end_sec": 48, "text": "들어갑니다"},
                 {"id": 2, "start_sec": 80, "end_sec": 82, "text": "멀리 떨어진 대사"}]
        tl = build_timeline(runs, lines, DEFAULT_PARAMS)
        self.assertEqual(len(tl.events), 1)
        ev = tl.events[0]
        self.assertEqual((ev.type, ev.team, ev.team_name, ev.game_time), ("kill", "blue", "T1", "03:24"))
        self.assertEqual(ev.players, {"killer": ["Doran"], "victims": ["Bin"]})
        self.assertEqual(ev.line_ids, [1])


class OpggTimelineTest(unittest.TestCase):
    """op.gg 킬·오브젝트를 화면 경기 시계로 영상 초로 옮긴다. 경기 0초 = 영상 100초."""

    def setUp(self):
        self.runs = [run("game_time", "", 100 + s, 101 + s, f"{s // 60:02d}:{s % 60:02d}") for s in range(0, 60)]
        self.runs += [run("game_time", "", 200 + s, 201 + s, f"{(100 + s) // 60:02d}:{(100 + s) % 60:02d}")
                      for s in range(0, 60)]  # 160~200초는 리플레이 (시계 없음)
        self.runs += [run("player_name", "blue_bot", 100, 260, "GEN Ruler"),
                      run("player_name", "red_jungle", 100, 260, "HLE Kanavi"),
                      run("kda", "blue_bot", 100, 131, "0/0/0"), run("kda", "blue_bot", 131, 260, "0/1/0"),
                      run("towers", "red", 100, 240, "0"), run("towers", "red", 240, 260, "1"),
                      run("team_kills", "red", 100, 131.5, "0"), run("team_kills", "red", 131.5, 260, "1")]
        self.ref = {"detailed": True,
                    "kills": [{"t": 30.0, "game_time": "00:30", "side": "red", "killer": "red_jungle",
                               "position": {"x": 1, "y": 2}},
                              {"t": 500.0, "game_time": "08:20", "side": "red", "killer": "red_jungle"}],  # 범위 밖
                    "objectives": [{"t": 80.0, "game_time": "01:20", "type": "dragon", "subtype": "water",
                                    "side": "blue", "killer": "blue_bot"}]}

    def test_events_from_opgg(self):
        from pipeline.game.timeline import build_timeline_opgg
        tl = build_timeline_opgg(self.runs, [], self.ref, DEFAULT_PARAMS)
        self.assertEqual(tl.source, "opgg")
        got = [(e.type, e.subtype, e.team, e.end_sec, e.source) for e in tl.events]
        self.assertEqual(got, [("kill", None, "red", 130.0, "opgg"),
                               ("objective", "dragon:water", "blue", 180.0, "opgg"),  # 리플레이 중 → 앞 시계로
                               ("tower", None, "red", 240, "ocr")])
        kill = tl.events[0]
        self.assertEqual(kill.players, {"killer": ["HLE Kanavi"], "victims": ["GEN Ruler"]})  # 피해자는 KDA 로
        self.assertEqual(kill.position, {"x": 1, "y": 2})
        self.assertEqual(tl.unmapped, [])
        # 동기화 검증: 화면 팀 킬 수는 1.5초 늦게 오름
        self.assertEqual(kill.ocr_lag_sec, 1.5)
        self.assertEqual((tl.sync["checked"], tl.sync["matched"], tl.sync["outliers"]), (1, 1, []))


if __name__ == "__main__":
    unittest.main()


class BannerTest(unittest.TestCase):
    def test_banner_objectives_and_dragon_team(self):
        from pipeline.game.timeline import banner_events, banner_subtype

        self.assertEqual([banner_subtype(t) for t in ("OCEAN DRAKE", "CHEMTECH DRAKE", "BARON NASHOR", "FANTASTEEL POSCO")],
                         ["dragon:water", "dragon:chemtech", "baron", None])
        runs = [{"tag": "dragon_icons", "slot": "blue", "start_sec": 0, "end_sec": 200, "text": "0.10"},
                {"tag": "dragon_icons", "slot": "red", "start_sec": 0, "end_sec": 105, "text": "0.00"},
                {"tag": "dragon_icons", "slot": "red", "start_sec": 105, "end_sec": 200, "text": "0.15"},
                {"tag": "banner", "slot": "", "start_sec": 100, "end_sec": 103, "text": "OCEAN DRAKE"},
                {"tag": "banner", "slot": "", "start_sec": 110, "end_sec": 112, "text": "OCEAN DRAKE"},  # 같은 배너 다시 → 하나
                {"tag": "banner", "slot": "", "start_sec": 300, "end_sec": 303, "text": "BARON NASHOR"}]
        ev = banner_events(runs, DEFAULT_PARAMS)
        self.assertEqual([(e[1], e[2], e[6]) for e in ev], [("dragon:water", "red", 100), ("baron", None, 300)])
