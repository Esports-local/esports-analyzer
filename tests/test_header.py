"""header 단계 조립 규칙 (op.gg·Data Dragon 응답을 흉내 낸다. 네트워크·영상 불필요)."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline import header
from pipeline.header import opgg
from pipeline.header.ddragon import pick_version

BLG = {"id": "1", "name": "Bilibili Gaming", "acronym": "BLG"}
T1 = {"id": "2", "name": "T1", "acronym": "T1"}


def match(year=2026, league="Worlds"):
    return {"id": "32706", "name": "Semifinal 1: BLG vs T1", "numberOfGames": 5, "matchVersion": "16.22.1",
            "beginAt": "2026-11-08T22:07:29.000Z", "homeTeam": BLG, "awayTeam": T1, "homeScore": 3, "awayScore": 1,
            "tournament": {"name": "Playoffs", "serie": {"year": year, "league": {"shortName": league}}},
            "videos": [{"url": "https://youtu.be/a", "type": "full-game", "game": {"id": "36891", "set": 2}}]}


GAME1 = {"id": "36890", "length": 1800, "winner": {"id": "1"},
         "teams": [{"side": "blue", "bans": [24], "team": {"id": "1"}}, {"side": "red", "bans": [], "team": {"id": "2"}}],
         "players": [{"side": "red", "position": "sup", "championId": "202", "player": {"nickName": "Keria"}}]}
GAME2 = {"id": "36891", "length": 1935, "winner": {"id": "2"},
         "teams": [{"side": "blue", "bans": [3], "team": {"id": "2"}}, {"side": "red", "bans": [1], "team": {"id": "1"}}],
         "players": [{"side": "red", "position": "jun", "championId": "246", "player": {"nickName": "XUN"}},
                     {"side": "blue", "position": "sup", "championId": "78", "player": {"nickName": "Keria"}},
                     {"side": "blue", "position": "coach", "championId": "1", "player": {"nickName": "?"}}]}
CHAMPS = {"78": {"id": "Poppy", "name": "Poppy", "ko": "뽀삐"}, "246": {"id": "Qiyana", "name": "Qiyana", "ko": "키아나"},
          "1": {"id": "Annie", "name": "Annie", "ko": "애니"}, "3": {"id": "Galio", "name": "Galio", "ko": "갈리오"},
          "24": {"id": "Jax", "name": "Jax", "ko": "잭스"}, "202": {"id": "Jhin", "name": "Jhin", "ko": "진"}}


def assemble(m=None, games=None):
    games = games or {1: opgg.parse_game(GAME1), 2: opgg.parse_game(GAME2)}
    return header.assemble("https://esports.op.gg/matches/32706", m or match(), games, 2, CHAMPS,
                           lambda cid: {"Q": f"{cid}-Q"}, "16.22.1")


class HeaderTest(unittest.TestCase):
    def test_parse_game(self):
        g = opgg.parse_game(GAME2)
        self.assertEqual((g["opgg_game_id"], g["winner"], g["sides"], g["bans"]),
                         (36891, 2, {"blue": 2, "red": 1}, {"blue": ["3"], "red": ["1"]}))
        self.assertEqual([p["role"] for p in g["players"]], ["jungle", "support"])  # coach 는 뺀다

    def test_assemble(self):
        h = assemble()
        self.assertEqual({k: h.match[k] for k in ("format", "set", "score", "video_url")},
                         {"format": "BO5", "set": 2, "score": {"BLG": 3, "T1": 1}, "video_url": "https://youtu.be/a"})
        self.assertEqual(h.teams["blue"], {"name": "T1", "full": "T1", "opgg_id": 2})
        self.assertEqual(h.players[0], {"side": "blue", "team": "T1", "role": "support", "name": "Keria",
                                        "champion": {"name": "Poppy", "ko": "뽀삐", "skills": {"Q": "Poppy-Q"}}})
        self.assertEqual(h.result, {"winner": "blue", "length_sec": 1935})
        s = h.series
        self.assertEqual((s["score_before"], s["previous"][0]["winner"]), ({"BLG": 1, "T1": 0}, "BLG"))
        self.assertEqual(s["previous"][0]["bans"], {"blue": [{"name": "Jax", "ko": "잭스"}], "red": []})
        # 피어리스: 1세트에 T1 이 고른 진은 못 고른다. 밴(잭스)은 넘어가지 않는다
        self.assertEqual(s["fearless_unavailable"], [{"name": "Jhin", "ko": "진", "set": 1, "team": "T1"}])

    def test_no_fearless_outside_leagues(self):
        s = assemble(match(year=2023)).series
        self.assertEqual((s["fearless"], s["fearless_unavailable"], len(s["previous"])), (False, [], 1))

    def test_run_keeps_going_on_error(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("pipeline.header.opgg.fetch_match", side_effect=ValueError("페이지에 경기 데이터가 없습니다")):
            path = header.run(Path("x.mp4"), Path(tmp), {"match": 32706, "set": 2})
            self.assertIn("ValueError", path.read_text(encoding="utf-8"))
            self.assertIn("ValueError", (path.parent / "opgg_timeline.json").read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            header.run(Path("x.mp4"), Path("."), {})  # match·set 없음

    def test_parse_timeline(self):
        game = {"detailedStats": True, "length": 2325,
                "teams": [{"side": "blue", "team": {"id": "416"}, "towerKills": 3, "frames": [{"gold": 2500, "timestamp": 60016}]},
                          {"side": "red", "team": {"id": "591"}, "towerKills": 9, "frames": []}],
                "objectKills": [{"teamId": "591", "objectType": "BARON_NASHOR", "objectSubType": None, "timestamp": 1877622},
                                {"teamId": "416", "objectType": "DRAGON", "objectSubType": "EARTH_DRAGON", "timestamp": 659064}],
                "players": [{"side": "blue", "position": "jun", "frames": [{"position": {"x": 130, "y": 401}, "timestamp": 0}],
                             "events": [{"type": "ELITE_MONSTER_KILL", "timestamp": 659064, "position": {"x": 9861, "y": 4415},
                                         "monsterType": "DRAGON"},
                                        {"type": "CHAMPION_KILL", "timestamp": 666323, "position": {"x": 10218, "y": 5639}}]},
                            {"side": "red", "position": "jun", "frames": [],
                             "events": [{"type": "ELITE_MONSTER_KILL", "timestamp": 1877400, "position": {"x": 5007, "y": 10471},
                                         "monsterType": "BARON_NASHOR"}]}]}
        t = opgg.parse_timeline(game)
        self.assertEqual(t["objectives"], [
            {"t": 659.064, "game_time": "10:59", "type": "dragon", "subtype": "earth", "side": "blue",
             "killer": "blue_jungle", "position": {"x": 9861, "y": 4415}},
            {"t": 1877.622, "game_time": "31:17", "type": "baron", "subtype": None, "side": "red",
             "killer": "red_jungle", "position": {"x": 5007, "y": 10471}}])
        self.assertEqual(t["kills"], [{"t": 666.323, "game_time": "11:06", "side": "blue", "killer": "blue_jungle",
                                       "position": {"x": 10218, "y": 5639}}])
        self.assertEqual((t["positions"]["blue_jungle"], t["gold"]["blue"], t["totals"]["red"]["towers"]),
                         ([[0.0, 130, 401]], [[60.016, 2500]], 9))

    def test_pick_version(self):
        self.assertEqual(pick_version(["16.23.1", "16.22.1"], "16.22.1"), "16.22.1")
        self.assertEqual(pick_version(["16.23.1", "16.21.2"], "16.21.1"), "16.21.2")
        with self.assertRaises(ValueError):
            pick_version([], "16.22.1")


if __name__ == "__main__":
    unittest.main()


class ThrottleTest(unittest.TestCase):
    def test_waits_between_requests(self):
        clock = [100.0]
        sleeps = []

        def sleep(s):
            sleeps.append(round(s, 3))
            clock[0] += s

        with mock.patch.object(opgg, "_last", 0.0), \
                mock.patch.object(opgg.time, "monotonic", lambda: clock[0]), \
                mock.patch.object(opgg.time, "sleep", sleep):
            opgg._throttle()          # 첫 요청은 바로
            clock[0] += 0.3
            opgg._throttle()          # 0.3초 뒤 → 0.7초 기다림
        self.assertEqual(sleeps, [0.7])
