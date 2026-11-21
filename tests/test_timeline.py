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


if __name__ == "__main__":
    unittest.main()
