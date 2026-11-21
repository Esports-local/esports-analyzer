"""받아쓰기 고유명사 교정 (pipeline/audio/terms.py)."""

import unittest

from pipeline.audio.terms import build_terms, correct_lines, correct_text, fit_particle, jamo

HEADER = {
    "players": [
        {"side": "red", "role": "jungle", "champion": {"ko": "나피리", "skills": {"Q": "단검 투척"}}},
        {"side": "blue", "role": "top", "champion": {"ko": "럼블", "skills": {}}},
        {"side": "red", "role": "top", "champion": {"ko": "갈리오", "skills": {}}},
        {"side": "blue", "role": "mid", "champion": {"ko": "라이즈", "skills": {}}},
    ],
    "bans": {"blue": [{"ko": "녹턴"}], "red": [{"ko": "진"}]},
    "series": {"fearless_unavailable": []},
}
EXTRA = [{"text": "한화생명", "kind": "team", "slot": "red"}, {"text": "딜라이트", "kind": "player", "slot": "red_support"}]


class TermsTest(unittest.TestCase):
    def setUp(self):
        self.terms = build_terms(HEADER, EXTRA)

    def fix(self, text):
        return correct_text(text, self.terms)[0]

    def test_champion_player_team(self):
        self.assertEqual(self.fix("나필이는 아래쪽에 있고요"), "나피리는 아래쪽에 있고요")
        self.assertEqual(self.fix("런블이 지나가면서"), "럼블이 지나가면서")
        self.assertEqual(self.fix("달리오는 위에"), "갈리오는 위에")
        self.assertEqual(self.fix("하나 생명 입장에서도"), "한화생명 입장에서도")
        self.assertEqual(self.fix("딜라잇."), "딜라이트.")

    def test_leaves_normal_words(self):
        for text in ("뭐 갈리오 분석기들이", "라이즈 쪽에", "진짜 잘했어요", "라인 수업 타이밍이", "들어오고 있어요"):
            self.assertEqual(self.fix(text), text)

    def test_fixes_recorded_and_reapplied_from_raw(self):
        once = correct_lines([{"id": 1, "text": "런블을 잡았다"}], self.terms)
        self.assertEqual(once[0], {"id": 1, "text": "럼블을 잡았다", "raw": "런블을 잡았다",
                                   "fixes": [{"from": "런블", "to": "럼블", "kind": "champion", "slot": "blue_top", "score": 0.83}]})
        self.assertEqual(correct_lines(once, self.terms), once)  # 다시 적용해도 원문에서 같은 결과

    def test_helpers(self):
        self.assertEqual(jamo("럼블"), "ㄹㅓㅁㅂㅡㄹ")
        self.assertEqual((fit_particle("나피리", "이는"), fit_particle("럼블", "를"), fit_particle("갈리오", "은")),
                         ("는", "을", "는"))
