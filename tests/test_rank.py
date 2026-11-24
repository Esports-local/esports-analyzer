"""장면형 LLM 결과 검사 (pipeline/shorts/rank.py). LLM 은 부르지 않는다 (가짜 client)."""

import json
import unittest

from pipeline.shorts import rank


def line(i, a, b, loud=0.5):
    return {"id": i, "from_sec": a, "to_sec": b, "text": f"대사{i}", "loud": loud}


def scene(sid, a, b, lines, events):
    return {"id": sid, "type": "scene", "fields": {"from_sec": a, "to_sec": b, "duration_sec": b - a,
                                                     "score": {}, "gold_diff": {"start": -200, "end": -1500}},
            "children": [{"id": f"cut-{sid}", "type": "cut",
                          "fields": {"from_sec": a, "to_sec": b, "events": events, "lines": lines}}]}


KILL = {"id": 1, "t": 120.0, "game_time": "15:10", "type": "kill", "killer": "HLE Kanavi (나피리)",
        "victims": ["GEN Ruler (루시안)"]}
INPUT = {"header": {"players": {"red_jungle": "HLE Kanavi (나피리)", "blue_bot": "GEN Ruler (루시안)",
                                "blue_support": "GEN Duro (밀리오)"}},
         "segments": [scene("c1", 100, 140, [line(1, 101, 105), line(2, 110, 114, 0.95), line(3, 116, 125, 0.97),
                                             line(4, 126, 130, 0.92), line(5, 131, 136, 0.6)], [KILL]),
                      scene("c2", 200, 230, [line(6, 201, 210), line(7, 212, 220, 0.8)],
                            [{"id": 2, "t": 215.0, "type": "objective", "subtype": "dragon:water"}])]}


class FakeClient:
    def __init__(self, answer):
        self.answer, self.calls = answer, 0

    def generate_json(self, system, user, schema, temperature=0.3):
        self.calls += 1
        return json.dumps(self.answer, ensure_ascii=False), {"promptTokenCount": 100}


class RankTest(unittest.TestCase):
    def test_good_answer_passes(self):
        ans = {"candidates": [{"id": "c1", "rank": 1, "start_line": 2, "end_line": 4, "titles": ["카나비 칼각", "15분 교환", "x"],
                               "reason": "외침"},
                              {"id": "c2", "rank": 2, "start_line": 6, "end_line": 7, "titles": ["바다 용"], "reason": "용"}]}
        res = rank.run(INPUT, client=FakeClient(ans))
        c1 = res["candidates"][0]
        self.assertEqual((c1["id"], c1["final"]["source"], c1["final"]["start_sec"], c1["final"]["end_sec"]),
                         ("c1", "llm", 109.5, 130.5))
        self.assertEqual(c1["final"]["title"], "카나비 칼각")
        self.assertTrue(all(c["result"] == "pass" for c in c1["checks"]))

    def test_fixes_cut_event_and_bad_titles(self):
        ans = {"candidates": [{"id": "c1", "rank": 1, "start_line": 3, "end_line": 4,  # 116초 시작 → 킬(120) 2초 여유는 됨
                               "titles": ["젠지 대 한화생명 15분 교환", "LCK 공식", "🔥"], "reason": ""},
                              {"id": "c2", "rank": 2, "start_line": 99, "end_line": 7, "titles": ["Ruler 2킬"], "reason": ""}]}
        res = rank.run(INPUT, client=FakeClient(ans))
        c1, c2 = res["candidates"]
        names = lambda c: {x["name"]: x["result"] for x in c["checks"]}
        self.assertEqual(names(c1)["title_rules"], "fixed")
        self.assertEqual(c1["final"]["title"], "Kanavi 킬")  # 다 버려져 자동 제목
        self.assertEqual(names(c2)["lines_in_scene"], "fixed")  # 없는 문장 번호 → 장면 전체
        self.assertEqual((c2["final"]["start_sec"], c2["final"]["end_sec"]), (200, 230))
        self.assertEqual(names(c2)["title_names"], "warn")     # Ruler 는 이 장면 이벤트에 없음
        self.assertEqual(names(c2)["title_counts"], "warn")    # 2킬인데 킬 0
        self.assertEqual(c1["final"]["source"], "llm+fixed")

    def test_event_cut_off_is_widened_and_shout_warned(self):
        ans = {"candidates": [{"id": "c1", "rank": 1, "start_line": 4, "end_line": 4, "titles": ["듀로 복수"], "reason": ""},
                              {"id": "c2", "rank": 2, "start_line": 6, "end_line": 7, "titles": ["용"], "reason": ""}]}
        c1 = rank.run(INPUT, client=FakeClient(ans))["candidates"][0]
        checks = {x["name"]: x for x in c1["checks"]}
        self.assertEqual(checks["covers_events"]["result"], "fixed")
        self.assertLessEqual(c1["final"]["start_sec"], 118.0)  # 킬 120초 2초 앞까지 넓힘
        self.assertEqual(checks["start_not_in_fight"]["result"], "warn")

    def test_no_llm_uses_rule_and_reuses_previous(self):
        res = rank.run(INPUT, allow_call=False)
        self.assertEqual([c["final"]["source"] for c in res["candidates"]], ["rule", "rule"])
        self.assertEqual(res["candidates"][0]["id"], "c1")  # 킬 + 외침이 용보다 위
        fake = FakeClient({"candidates": []})
        first = rank.run(INPUT, client=fake)
        rank.run(INPUT, previous=first, client=fake)
        self.assertEqual(fake.calls, 1)  # 같은 입력이면 다시 부르지 않음

    def test_bad_ranks_fall_back_to_rule_order(self):
        ans = {"candidates": [{"id": "c1", "rank": 1, "start_line": 2, "end_line": 4, "titles": ["a"], "reason": ""},
                              {"id": "c2", "rank": 1, "start_line": 6, "end_line": 7, "titles": ["b"], "reason": ""}]}
        res = rank.run(INPUT, client=FakeClient(ans))
        self.assertEqual([c["final"]["rank"] for c in res["candidates"]], [1, 2])
        self.assertEqual(res["candidates"][0]["final"]["source"], "llm+fixed")

    def test_end_extends_through_shouts(self):
        # 끝 문장 3(116~125) 뒤 4 가 외침, 5 는 정리 → 5 끝(136) + 0.5
        ans = {"candidates": [{"id": "c1", "rank": 1, "start_line": 2, "end_line": 3, "titles": ["카나비 칼각"], "reason": ""},
                              {"id": "c2", "rank": 2, "start_line": 6, "end_line": 7, "titles": ["바다 용"], "reason": ""}]}
        c1 = rank.run(INPUT, client=FakeClient(ans))["candidates"][0]
        checks = {x["name"]: x for x in c1["checks"]}
        self.assertEqual(checks["end_extend"]["result"], "fixed")
        self.assertEqual(c1["final"]["end_sec"], 136.5)
        self.assertNotIn("end_not_in_fight", checks)
        self.assertEqual(c1["final"]["source"], "llm+fixed")

    def test_end_extend_stops_after_calm_line(self):
        lines = [line(1, 0, 4), line(2, 5, 8, 0.95), line(3, 9, 11, 0.4), line(4, 12, 14, 0.95)]
        self.assertEqual(rank._extend_end(lines, 4.5, 100), (11.5, [2, 3]))
        self.assertEqual(rank._extend_end(lines, 8.5, 100), (8.5, []))   # 뒤가 조용하면 그대로
        self.assertEqual(rank._extend_end(lines, 4.5, 9.0), (8.5, [2]))  # 장면 끝 뒤 문장은 더하지 않음
        self.assertEqual(rank._extend_end(lines, 4.5, 7.0), (8.5, [2]))  # 장면 끝을 걸친 문장은 끝까지
        self.assertEqual(rank._extend_end(lines, 4.5, 4.5), (4.5, []))   # 장면 끝 뒤에서 시작하는 외침은 안 넣음

    def test_title_result_drops_wrong_winner(self):
        inp = json.loads(json.dumps(INPUT))
        inp["header"].update(teams={"blue": "GEN (Gen.G, 젠지)", "red": "HLE (Hanwha Life Esports, 한화생명)"},
                             winner_side="red")
        ans = {"candidates": [{"id": "c1", "rank": 1, "start_line": 2, "end_line": 4,
                               "titles": ["34분 젠지 대승", "카나비 칼각", "한화생명 승리"], "reason": ""},
                              {"id": "c2", "rank": 2, "start_line": 6, "end_line": 7, "titles": ["한화생명 패배"], "reason": ""}]}
        c1, c2 = rank.run(inp, client=FakeClient(ans))["candidates"]
        self.assertEqual(c1["final"]["titles"], ["카나비 칼각", "한화생명 승리"])
        self.assertEqual({x["name"]: x["result"] for x in c1["checks"]}["title_result"], "fixed")
        self.assertNotIn("한화생명 패배", c2["final"]["titles"])  # 이긴 팀이 졌다고 함 → 자동 제목

    def test_llm_input_header_has_winner(self):
        from pipeline.shorts import llm_input
        data = {"header": {"game": {"winner": "red"}, "context": {"teams": {
            "blue": {"name": "GEN", "full": "Gen.G"}, "red": {"name": "HLE", "full": "Hanwha Life Esports"}}}},
            "segments": []}
        h = llm_input.build(data, [], {"GEN": "젠지", "HLE": "한화생명"})["header"]
        self.assertEqual(h["teams"]["blue"], "GEN (Gen.G, 젠지)")
        self.assertEqual((h["winner_side"], h["winner"]), ("red", "HLE (Hanwha Life Esports, 한화생명)"))


if __name__ == "__main__":
    unittest.main()


class QueryTest(unittest.TestCase):
    PLAYERS = {"blue_bot": "GEN Ruler (루시안)", "red_jungle": "HLE Kanavi (나피리)"}

    def test_parse_cleans_values(self):
        from pipeline.shorts import query
        fake = FakeClient({"killer": "blue_bot", "type": "kill", "team": "", "tag": "pentakill",
                           "title": "룰러 킬 모음", "note": ""})
        res = query.parse("룰러 킬 모음", self.PLAYERS, client=fake)
        self.assertEqual(res, {"filter": {"killer": "blue_bot", "type": "kill"}, "title": "룰러 킬 모음",
                               "note": None, "error": None})
        bad = query.clean({"killer": "blue_top", "type": "death", "title": "아주아주아주 긴 제목입니다"}, self.PLAYERS)
        self.assertEqual((bad["filter"], bad["title"]), ({}, None))  # 목록에 없는 슬롯·종류, 10자 넘는 제목

    def test_endpoint(self):
        from unittest import mock
        from fastapi.testclient import TestClient
        from app.main import app
        with mock.patch("pipeline.shorts.query.parse", return_value={"filter": {"killer": "blue_bot"}, "title": "룰러 킬",
                                                                      "note": None, "error": None}):
            res = TestClient(app).post("/shorts/query", json={"text": "룰러 킬", "players": self.PLAYERS})
        self.assertEqual((res.status_code, res.json()["filter"]), (200, {"killer": "blue_bot"}))
