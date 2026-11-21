"""segments 전환 찾기·나누기 (TransNetV2 없이 확률·프레임을 만들어 넣는다)."""

import unittest

import numpy as np

from pipeline.segments import (DEFAULT_PARAMS, build_segments, classify, classify_gradual, find_transition_spans,
                               merge_graphic_wipes)


def gradual_frames(kind: str, n: int = 12) -> np.ndarray:
    """앞 장면(검정 아님 어두운 회색) → 뒤 장면(밝은 색) 점진 전환. 0·n-1 은 앞뒤 장면 그대로."""
    a, b = np.full((27, 48, 3), 40, np.float32), np.full((27, 48, 3), 220, np.float32)
    out = []
    for i in range(n):
        x = min(1.0, max(0.0, (i - 1) / (n - 3)))
        if kind == "dissolve":
            f = a * (1 - x) + b * x
        else:  # wipe: 왼쪽부터 오른쪽으로
            f = a.copy()
            f[:, : int(48 * x)] = 220
        out.append(f.astype(np.uint8))
    return np.stack(out)


class SegmentsTest(unittest.TestCase):
    def test_dissolve_vs_wipe(self):
        self.assertEqual(classify_gradual(gradual_frames("dissolve"), 1, 10, 0.35), "dissolve")
        self.assertEqual(classify_gradual(gradual_frames("wipe"), 1, 10, 0.35), "wipe")

    def test_spans_and_segments(self):
        n, fps = 300, 30.0
        single, many = np.zeros(n), np.zeros(n)
        single[99] = 0.9                    # 컷: 100번 프레임부터 새 장면
        many[200:210] = 0.8                 # 점진 전환
        luma = np.full(n, 100.0)
        spans = find_transition_spans(single, many, luma, fps, DEFAULT_PARAMS)
        self.assertEqual(spans, [(99, 99, False), (200, 209, True)])
        frames = np.zeros((n, 27, 48, 3), np.uint8)
        spans = merge_graphic_wipes(spans, np.zeros(n - 1), fps, DEFAULT_PARAMS)
        trs = [classify(frames, luma, a, b, k, i, fps, DEFAULT_PARAMS) for i, (a, b, k) in enumerate(spans, 1)]
        self.assertEqual([(t.kind, t.start_sec, t.end_sec) for t in trs], [("cut", 3.333, 3.333), ("dissolve", 6.667, 7.0)])
        trs, segs = build_segments(trs, n / fps, 0.3)
        self.assertEqual([(s.start_sec, s.end_sec, s.transition_in, s.transition_out) for s in segs],
                         [(0.0, 3.333, None, 1), (3.333, 6.667, 1, 2), (7.0, 10.0, 2, None)])

    def test_graphic_wipe_merges_two_cuts(self):
        motion = np.zeros(300)
        motion[100:124] = 30   # 컷 99 와 123 사이 화면이 계속 움직임 (로고 애니메이션)
        spans = [(99, 99, False), (123, 123, False), (200, 200, False), (220, 220, False)]  # 200~220 은 멈춘 화면
        self.assertEqual(merge_graphic_wipes(spans, motion, 30.0, DEFAULT_PARAMS),
                         [(99, 123, "wipe"), (200, 200, "cut"), (220, 220, "cut")])
