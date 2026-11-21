"""레이아웃 자동 선택: header 리그 → configs/lol/layouts.json."""

import json
import tempfile
import unittest
from pathlib import Path

from pipeline.screen.layout import load, resolve


def write_header(work: Path, league: str | None, error: str | None = None):
    result = {"match": {"league": league} if league else {}, "error": error}
    (work / "work").mkdir(exist_ok=True)
    (work / "work" / "header.json").write_text(json.dumps({"result": result}), encoding="utf-8")


class LayoutsTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.work = Path(tmp.name)

    def test_league_picks_layout(self):
        write_header(self.work, "LCK")
        self.assertEqual(resolve("auto", self.work), "configs/lol/lck_2026.json")
        self.assertIn("screens", load(resolve("auto", self.work)))

    def test_default_without_header_or_on_error_or_unknown(self):
        self.assertEqual(resolve("auto", self.work), "configs/lol/worlds_2026.json")
        self.assertNotIn("screens", load(resolve("auto", self.work)))
        write_header(self.work, "LCK", error="ConnectError")
        self.assertEqual(resolve("auto", self.work), "configs/lol/worlds_2026.json")
        write_header(self.work, "LEC")
        self.assertEqual(resolve("auto", self.work), "configs/lol/worlds_2026.json")

    def test_explicit_path_kept(self):
        write_header(self.work, "LCK")
        self.assertEqual(resolve("configs/x.json", self.work), "configs/x.json")
        self.assertIsNone(resolve(None, self.work))

    def test_every_listed_file_exists(self):
        conf = json.loads(Path("configs/lol/layouts.json").read_text(encoding="utf-8"))
        for path in [*conf["leagues"].values(), conf["default"]]:
            self.assertIn("rois", load(path), path)
