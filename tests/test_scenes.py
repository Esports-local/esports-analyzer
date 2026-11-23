"""숏폼 후보 장면 묶기 테스트 (analyzer.json 모양을 작게 만들어 씀)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline.shorts import scenes
from pipeline.shorts.scenes import candidates, group


def cut(i, a, b, kind="live", lines=(), events=()):
    return {"id": f"cut-{i}", "type": "cut",
            "fields": {"from_sec": a, "to_sec": b, "kind": kind, "lines": list(lines), "events": list(events)}}


def line(i, a, b):
    return {"id": i, "from_sec": a, "to_sec": b, "text": f"대사{i}"}


def data(*cuts):
    return {"segments": [{"type": "phase", "children": list(cuts)}]}


class ScenesTest(unittest.TestCase):
    def test_line_across_boundary_joins_cuts(self):
        cuts = [cut(1, 0, 10, lines=[line(1, 8, 12)]), cut(2, 10, 20), cut(3, 20, 30, lines=[line(2, 21, 25)])]
        self.assertEqual([[c["id"] for c in s] for s in group(cuts)], [["cut-1", "cut-2"], ["cut-3"]])

    def test_small_overlap_and_kind_change_split(self):
        cuts = [cut(1, 0, 10, lines=[line(1, 5, 10.2)]), cut(2, 10, 20, lines=[line(2, 18, 22)]),
                cut(3, 20, 30, kind="replay")]
        self.assertEqual(len(group(cuts)), 3)

    def test_candidates_skip_minor_and_replay(self):
        kill = {"id": 1, "t": 5, "type": "kill", "tags": []}
        grubs = {"id": 2, "t": 15, "type": "objective", "subtype": "grubs"}
        tower = {"id": 3, "t": 25, "type": "tower"}
        drake = {"id": 4, "t": 35, "type": "objective", "subtype": "dragon:water"}
        replay_kill = {"id": 5, "t": 45, "type": "kill"}
        d = data(cut(1, 0, 10, events=[kill]), cut(2, 10, 20, events=[grubs]), cut(3, 20, 30, events=[tower]),
                 cut(4, 30, 40, events=[drake]), cut(5, 40, 50, kind="replay", events=[replay_kill]))
        got = candidates(d)
        self.assertEqual([c["cuts"] for c in got], [["cut-1"], ["cut-4"]])
        self.assertEqual(got[0]["id"], "c1")

    def test_no_kind_uses_game_range(self):
        kill = {"id": 1, "t": 5, "type": "kill"}
        late = {"id": 2, "t": 45, "type": "kill"}
        d = data(cut(1, 0, 10, kind=None, events=[kill]), cut(2, 40, 50, kind=None, events=[late]))
        self.assertEqual(len(candidates(d)), 2)  # 경기 구간 정보 없음 → 모두 live
        d["header"] = {"game": {"start_sec": 30, "end_sec": 60}}
        self.assertEqual([c["cuts"] for c in candidates(d)], [["cut-2"]])

    def test_clip_span_follows_loud_lines(self):
        from pipeline.shorts.scenes import DEFAULT_PARAMS, clip_span
        L = lambda a, b, loud: {"from_sec": a, "to_sec": b, "_loud": loud}
        lines = [L(80, 85, 0.5), L(86, 92, 0.9), L(93, 99, 0.95), L(101, 104, 0.9), L(105, 109, 0.6)]
        # 이벤트 100초: 앞은 0.85 이상 문장, 뒤는 104초까지 (0.6 문장 앞에서 멈춤)
        self.assertEqual(clip_span(100, lines, 0, 200, DEFAULT_PARAMS), (93, 104))  # 86초 문장은 12초 앞(88)보다 일러서 안 넣음
        # 장면 범위가 좁으면 그 안으로
        self.assertEqual(clip_span(100, lines, 97, 103, DEFAULT_PARAMS), (97, 103))

    def test_event_clips_with_replay(self):
        from pipeline.shorts.scenes import event_clips
        kill = {"id": 1, "t": 15, "type": "kill", "team": "red", "killer": "red_jungle"}
        d = data(cut(1, 10, 20, events=[kill]), cut(2, 30, 40, kind="replay"))
        clips = event_clips(d)
        self.assertEqual((clips[0]["start_sec"], clips[0]["end_sec"], clips[0]["scene"]), (11, 18, "c1"))
        self.assertEqual(clips[0]["replay"], {"start_sec": 30, "end_sec": 40, "cuts": ["cut-2"]})

    def test_run_writes_stage(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            (work / "work").mkdir()
            (work / "work" / "context.json").write_text("{}", encoding="utf-8")
            tree = data(cut(1, 0, 10, events=[{"id": 1, "t": 5, "type": "kill"}]), cut(2, 10, 20))
            with mock.patch("pipeline.game.bundle.build", return_value=tree):
                path = scenes.run(Path("game.mp4"), work, {"use_llm": False})
            out = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(out["stage"], "shorts")
            self.assertEqual(out["result"]["scenes"], 2)
            self.assertEqual(out["result"]["candidates"][0]["cuts"], ["cut-1"])

    def test_run_needs_context(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(FileNotFoundError):
            scenes.run(Path("game.mp4"), Path(d))



class LlmInputTest(unittest.TestCase):
    def test_build_keeps_shape_and_slims(self):
        from pipeline.shorts.llm_input import build
        kill = {"id": 1, "t": 12, "type": "kill", "team": "red", "killer": "red_jungle", "victims": ["blue_bot"],
                "assists": [], "tags": [], "source": ["x"], "lines": [{"id": 1}]}
        c1 = cut(1, 0, 10, lines=[line(1, 1, 2)])
        c1["fields"]["state"] = {"kills": {"blue": 0, "red": 0}, "gold_diff": 0}
        c2 = cut(2, 10, 20, lines=[line(2, 11, 13.5)], events=[kill])
        c2["fields"].update(state={"kills": {"blue": 0, "red": 1}, "gold_diff": -300},
                            signals={"loudness": [0.1, 0.9, 0.95, 0.2], "gold_diff": [0, 0, -300, -300]},
                            reference=[{"type": "kill"}])
        d = data(c1, c2)
        d["header"] = {"range": {"from_sec": 0}, "players": {
            "red_jungle": {"team": "HLE", "name": "Kanavi", "champion": "나피리"},
            "blue_bot": {"team": "GEN", "name": "Ruler", "champion": "루시안"}}}
        cand = candidates(d)
        out = build(d, cand)
        scene = out["segments"][0]
        self.assertEqual(scene["type"], "scene")
        self.assertEqual(scene["fields"]["score"], {"start": {"blue": 0, "red": 0}, "end": {"blue": 0, "red": 1}})
        f = scene["children"][0]["fields"]
        self.assertNotIn("reference", f)
        self.assertNotIn("signals", f)
        self.assertEqual(f["events"][0]["killer"], "HLE Kanavi (나피리)")
        self.assertEqual(f["lines"][0]["loud"], 0.95)  # 11~13.5초 → 배열 1~3번 중 최댓값
        self.assertIn("scene", out["header"]["segment_types"])

    def test_loud_reads_next_cut_and_step(self):
        from pipeline.shorts.llm_input import build
        kill = {"id": 1, "t": 15, "type": "kill"}
        a = cut(1, 0, 10, lines=[line(1, 8, 14), line(2, 0, 1)])
        a["fields"]["signals"] = {"start_sec": 0.0, "step_sec": 0.5, "loudness": [0.1] * 21}
        b = cut(2, 10, 20, lines=[line(3, 15, 16)], events=[kill])
        b["fields"]["signals"] = {"start_sec": 10.0, "step_sec": 0.5, "loudness": [0.2] * 4 + [0.95] + [0.2] * 16}
        d = data(a, b)
        out = build(d, candidates(d))
        lines = {l["id"]: l["loud"] for c in out["segments"][0]["children"] for l in c["fields"]["lines"]}
        self.assertEqual(lines[1], 0.95)  # 8~14초 문장: 다음 컷 12초(0.5초 간격 4번째) 최고점까지
        self.assertEqual(lines[2], 0.1)

    def test_unknown_cut_skipped(self):
        from pipeline.shorts.llm_input import build
        d = data(cut(1, 0, 10))
        out = build(d, [{"id": "c1", "cuts": ["cut-9"], "start_sec": 0, "end_sec": 10, "duration_sec": 10}])
        self.assertEqual(out["segments"], [])


if __name__ == "__main__":
    unittest.main()
