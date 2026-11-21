"""segments 단계: 장면 전환(cut · dissolve · wipe)으로 영상을 나누고, 구간마다 화면 종류를 붙인다.

1. 전환 찾기 — TransNetV2 (CPU). 전체 프레임을 원래 fps 그대로 48x27 로 넣는다
   (10fps 로 줄이면 옵저버 하드컷·리플레이 전환을 절반 놓쳤다, 2026-11-21 LCK 결승 3분 비교).
   - single: 컷 지점 확률, many: 전환 구간에 속한 프레임 확률
   - 점진 전환: many > many_threshold 가 gradual_min_frames 이상. 양끝은 many > many_low 까지 넓힌다
   - cut: single > single_threshold 인 지점 중 점진 전환 밖 (한 프레임)
   - 사이가 아주 짧거나(min_segment_sec 미만) 짧은(max_black_gap_sec 이하) 완전한 검정이면 하나로 합친다
2. 중계 그래픽 전환(리플레이 들어가고 나올 때 로고가 쓸고 지나가는 0.5~1초 애니메이션)은 TransNetV2 가
   앞뒤 컷 두 개로 잡는다. 두 컷이 graphic_max_sec 안이고 그 사이 화면이 계속 크게 움직이면(motion) 하나의 wipe 로 합친다
   (merge_graphic_wipes). 그 사이가 멈춘 화면이면 옵저버가 연달아 자른 것이라 그대로 둔다.
3. 점진 전환을 dissolve / wipe 로 나눈다 (모델 없음, classify_gradual).
   화면을 칸으로 나눠 칸마다 앞 장면 → 뒤 장면으로 바뀐 시점을 본다.
   모든 칸이 거의 같이 바뀌면 dissolve(페이드 포함, 검정을 거치면 black), 칸마다 바뀐 시점이 퍼져 있으면 wipe
   (한쪽부터 쓸고 지나가는 전환, 중계 그래픽 전환 포함).
4. 전환 사이가 segment. 두 cut 사이가 min_segment_sec 미만이면(플래시 등) 앞 cut 을 지워 합친다.
5. 화면 종류 — 레이아웃 파일 screens 규칙이 있으면(지금은 LCK) sample_sec 마다 화면을 판별해
   (pipeline/screen/rules.py, 정해진 자리 글자 OCR) segment 마다 다수결로 kind 를 붙이고 경기 구간(game)을 정한다.

시간: 프레임 i 는 [i/fps, (i+1)/fps). fps 필터로 CFR 에 맞춰 읽으므로 VFR 영상도 시각이 맞다.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np

from pipeline import media
from pipeline.io import save_stage, stage_path
from pipeline.screen.layout import load as load_layout
from pipeline.screen.layout import resolve
from pipeline.media import fps_of, probe, read_frames, video_stream
from pipeline.schema import Segment, SegmentsResult, Transition

log = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "layout": "auto",           # 레이아웃 파일 (screens 규칙). auto 면 header 리그로 고름, null 이면 화면 종류 안 붙임
    "single_threshold": 0.5,
    "many_threshold": 0.5,
    "many_low": 0.2,
    "gradual_min_frames": 3,
    "fade_black_luma": 20.0,    # 점진 전환 안에 이보다 어두운 프레임이 있으면 black=True
    "black_gap_luma": 10.0,     # 두 전환 사이가 이보다 어두우면 (거의 완전한 검정) 하나로 합칠 후보
    "max_black_gap_sec": 2.0,   # 그 검정 구간이 이 길이 이하일 때만 합친다
    "min_segment_sec": 0.3,
    "wipe_spread": 0.35,        # 칸마다 바뀐 시점의 퍼짐(전환 길이 대비)이 이보다 크면 wipe
    "graphic_max_sec": 1.2,     # 이 안에 붙은 컷 두 개 + 사이가 계속 움직이면 그래픽 전환(wipe) 하나
    "graphic_motion": 8.0,      # 그 사이 프레임 차이(48x27 평균 절대값, 0~255) 평균이 이 이상이면 움직임
    "chunk_frames": 9000,       # TransNetV2 를 이만큼씩 나눠 돌리며 진행 로그를 남긴다
}
CHUNK_PAD = 50  # 조각 경계에서 TransNetV2 문맥(앞뒤 25프레임)을 잃지 않도록 겹쳐 읽는 프레임 수
GRID = (3, 6)   # wipe 판정용 칸 (세로, 가로). 48x27 프레임에서 칸 하나는 8x9 픽셀


def run(video: Path, work_dir: Path, params: dict | None = None) -> Path:
    """segments 단계를 실행한다. Returns: work/segments.json 경로."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    fps = fps_of(video_stream(probe(video)))
    t0 = time.monotonic()
    frames = read_frames(video, 48, 27, fps)
    log.info("segments: 프레임 %d장 읽음 (%.0f초)", len(frames), time.monotonic() - t0)
    luma = frame_luma(frames)
    single, many = predict(frames, p["chunk_frames"])
    duration = len(frames) / fps
    spans = find_transition_spans(single, many, luma, fps, p)
    spans = merge_graphic_wipes(spans, frame_motion(frames), fps, p)
    transitions = [classify(frames, luma, a, b, kind, i, fps, p) for i, (a, b, kind) in enumerate(spans, 1)]
    del frames
    transitions, segments = build_segments(transitions, duration, p["min_segment_sec"])
    if media.START_SEC:  # 일부 구간만 읽었으면 원본 시각으로
        transitions, segments = shift(transitions, media.START_SEC), shift(segments, media.START_SEC)
    log.info("segments: 전환 %d개 %s", len(transitions),
             {k: sum(1 for t in transitions if t.kind == k) for k in ("cut", "dissolve", "wipe")})

    p["fps"] = fps
    p["layout"] = resolve(p["layout"], work_dir)
    screens = load_layout(p["layout"]).get("screens") if p["layout"] else None
    result = SegmentsResult(layout=p["layout"], transitions=transitions, segments=segments)
    if screens:
        result = label_screens(video, work_dir, result, screens)
    return save_stage(work_dir, "segments", result, p)


def label_screens(video: Path, work_dir: Path, result: SegmentsResult, screens: dict) -> SegmentsResult:
    """sample_sec 마다 화면 종류를 판별해 segment 에 붙이고 경기 구간을 정한다."""
    from pipeline.media import iter_frames
    from pipeline.screen.roi_ocr import OcrReader
    from pipeline.screen.rules import classify_frame, game_range, label_segments, load_rules

    rules, default, step = load_rules(screens)
    reader = OcrReader()
    t0 = time.monotonic()
    samples = [{"t": round(media.START_SEC + i * step, 3), **classify_frame(f, rules, default, reader)}
               for i, f in enumerate(iter_frames(video, 1920, 1080, 1 / step))]
    labeled = label_segments([s.model_dump() for s in result.segments], samples, default)
    counts = {k: sum(1 for s in samples if s["kind"] == k) for k in sorted({s["kind"] for s in samples})}
    game = game_range(samples, step, _header(work_dir))
    log.info("segments: 화면 %d장 판별 %s (%.0f초), 경기 구간 %s", len(samples), counts, time.monotonic() - t0, game)
    return result.model_copy(update={"segments": [Segment(**s) for s in labeled], "sample_sec": step,
                                     "samples": counts, "game": game})


def shift(items: list, off: float) -> list:
    """Transition · Segment 목록의 시각에 off 초를 더한다."""
    return [x.model_copy(update={"start_sec": round(x.start_sec + off, 3), "end_sec": round(x.end_sec + off, 3)})
            for x in items]


def _header(work_dir: Path) -> dict | None:
    path = stage_path(work_dir, "header")
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8")).get("result") or {}
    return None if result.get("error") else result


def frame_luma(frames: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """프레임별 평균 밝기(0~255). 한 번에 float 로 바꾸면 메모리가 4배라 chunk 씩 계산한다."""
    weights = np.array([0.299, 0.587, 0.114], np.float32)
    out = np.empty(len(frames), np.float32)
    for i in range(0, len(frames), chunk):
        part = frames[i : i + chunk]
        out[i : i + len(part)] = (part.reshape(len(part), -1, 3) @ weights).mean(axis=1)
    return out


def frame_motion(frames: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """motion[i] = 프레임 i 와 i+1 의 평균 절대 차이 (0~255). 길이 N-1."""
    out = np.empty(max(0, len(frames) - 1), np.float32)
    for i in range(0, len(out), chunk):
        a = frames[i : i + chunk + 1].astype(np.int16)
        out[i : i + len(a) - 1] = np.abs(np.diff(a, axis=0)).mean(axis=(1, 2, 3))
    return out


def merge_graphic_wipes(spans: list[tuple[int, int, bool]], motion: np.ndarray, fps: float, p: dict
                        ) -> list[tuple[int, int, str]]:
    """(a, b, gradual) → (a, b, kind). kind: cut | gradual | wipe.

    붙어 있는 컷 두 개 사이가 graphic_max_sec 이하이고, 사이 프레임 차이 평균이 graphic_motion 이상이면
    중계 그래픽 전환 하나(wipe)로 합친다.
    """
    out: list[tuple[int, int, str]] = []
    for a, b, g in spans:
        kind = "gradual" if g else "cut"
        if out and kind == "cut" and out[-1][2] == "cut":
            pa, _, _ = out[-1]
            between = motion[pa + 1 : a]
            if 0 < a - pa <= p["graphic_max_sec"] * fps and len(between) and between.mean() >= p["graphic_motion"]:
                out[-1] = (pa, b, "wipe")
                continue
        out.append((a, b, kind))
    return out


def predict(frames: np.ndarray, chunk: int) -> tuple[np.ndarray, np.ndarray]:
    """TransNetV2 로 프레임마다 전환 확률 (single, many). chunk 프레임씩 앞뒤를 겹쳐 돌리며 진행을 로그로 남긴다.

    Args:
        frames: (N, 27, 48, 3) uint8 RGB. TransNetV2 는 이 크기만 받는다.
    """
    import torch
    from transnetv2_pytorch import TransNetV2

    model = TransNetV2(device="cpu")
    model.eval()
    n = len(frames)
    single, many = np.zeros(n, np.float32), np.zeros(n, np.float32)
    t0 = time.monotonic()
    with torch.no_grad():
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            a, b = max(0, start - CHUNK_PAD), min(n, end + CHUNK_PAD)
            s, m = model.predict_frames(torch.from_numpy(np.ascontiguousarray(frames[a:b])), quiet=True)
            single[start:end] = s.numpy()[start - a : end - a]
            many[start:end] = m.numpy()[start - a : end - a]
            spent = time.monotonic() - t0
            log.info("segments: TransNetV2 %d/%d (%.0f%%), %.0f초, 남은 시간 약 %.0f초",
                     end, n, end / n * 100, spent, spent / end * (n - end))
    return single, many


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """bool 배열에서 True 가 이어진 (시작, 끝) 인덱스 목록 (끝 포함)."""
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def find_transition_spans(
    single: np.ndarray, many: np.ndarray, luma: np.ndarray, fps: float, p: dict
) -> list[tuple[int, int, bool]]:
    """전환 확률 → (a, b, gradual) 목록. a, b 는 양끝 포함 프레임 번호, 컷이면 a == b (컷 직전 프레임).

    겹치거나, 사이가 min_segment_sec 미만인데 점진 전환이 끼었거나, 사이가 짧은 완전한 검정(페이드 아웃 → 검정 →
    페이드 인)이면 앞뒤 전환을 하나로 합친다. 검정 길이를 제한하지 않으면 어두운 장면 전체가 전환으로 사라진다.
    """
    n = len(many)
    spans: list[tuple[int, int, bool]] = []
    for a, b in _runs(many > p["many_threshold"]):
        if b - a + 1 < p["gradual_min_frames"]:
            continue
        while a > 0 and many[a - 1] > p["many_low"]:
            a -= 1
        while b < n - 1 and many[b + 1] > p["many_low"]:
            b += 1
        spans.append((a, b, True))
    for a, b in _runs(single > p["single_threshold"]):
        peak = a + int(np.argmax(single[a : b + 1]))
        if not any(ga <= peak <= gb for ga, gb, _ in spans):
            spans.append((peak, peak, False))
    spans.sort()

    min_gap = p["min_segment_sec"] * fps
    max_black_gap = p["max_black_gap_sec"] * fps
    merged: list[tuple[int, int, bool]] = []
    for a, b, g in spans:
        if merged:
            pa, pb, pg = merged[-1]
            gap = a - pb - 1
            black_gap = 0 < gap <= max_black_gap and luma[pb + 1 : a].max() < p["black_gap_luma"]
            if gap <= 0 or black_gap or (gap < min_gap and (g or pg)):
                merged[-1] = (pa, max(pb, b), True)
                continue
        merged.append((a, b, g))
    return merged


def classify_gradual(frames: np.ndarray, a: int, b: int, spread_limit: float) -> str:
    """점진 전환 [a, b] 가 dissolve 인지 wipe 인지.

    앞 장면 A = frames[a-1], 뒤 장면 B = frames[b+1]. 칸마다 프레임이 A 보다 B 에 가까워지는 첫 시점을 찾고,
    그 시점들의 퍼짐(최대 - 최소)이 전환 길이의 spread_limit 배보다 크면 wipe.
    """
    n = len(frames)
    if b - a < 2 or a == 0 or b >= n - 1:
        return "dissolve"
    gh, gw = GRID
    h, w = frames.shape[1] // gh, frames.shape[2] // gw
    span = frames[a - 1 : b + 2].astype(np.float32)
    blocks = span[:, : gh * h, : gw * w].reshape(len(span), gh, h, gw, w, 3).mean(axis=(2, 4))  # (T, gh, gw, 3)
    first, last = blocks[0], blocks[-1]
    d_a = np.abs(blocks - first).sum(-1)
    d_b = np.abs(blocks - last).sum(-1)
    changed = np.abs(last - first).sum(-1) > 30  # 앞뒤가 거의 같은 칸(배경이 같은 곳)은 판단에서 뺀다
    if changed.sum() < 3:
        return "dissolve"
    switch = (d_b < d_a).argmax(axis=0)[changed]  # 칸마다 B 에 더 가까워진 첫 프레임
    spread = (switch.max() - switch.min()) / max(1, len(span) - 1)
    return "wipe" if spread > spread_limit else "dissolve"


def classify(
    frames: np.ndarray, luma: np.ndarray, a: int, b: int, kind: str, tid: int, fps: float, p: dict
) -> Transition:
    """전환 구간 하나 → Transition.

    - cut: start == end == (a + 1) / fps (컷 뒤 첫 프레임 시작)
    - wipe(그래픽 전환, 컷 두 개를 합친 것): (a + 1) / fps ~ (b + 1) / fps
    - gradual: [a / fps, (b + 1) / fps]. 검정을 거치면 dissolve(black), 아니면 classify_gradual 로 dissolve / wipe
    """
    if kind == "cut":
        t = round((a + 1) / fps, 3)
        return Transition(id=tid, kind="cut", start_sec=t, end_sec=t, black=False)
    if kind == "wipe":
        return Transition(id=tid, kind="wipe", start_sec=round((a + 1) / fps, 3), end_sec=round((b + 1) / fps, 3))
    black = bool(luma[a : b + 1].min() < p["fade_black_luma"])
    kind = "dissolve" if black else classify_gradual(frames, a, b, p["wipe_spread"])
    return Transition(id=tid, kind=kind, start_sec=round(a / fps, 3), end_sec=round((b + 1) / fps, 3), black=black)


def build_segments(
    transitions: list[Transition], duration: float, min_sec: float
) -> tuple[list[Transition], list[Segment]]:
    """전환 목록 → segment 목록. 두 cut 사이가 min_sec 미만이면 앞 cut 을 지운다. 전환 구간은 segment 에 안 넣는다."""
    kept: list[Transition] = []
    for t in transitions:
        prev_end = kept[-1].end_sec if kept else 0.0
        if kept and t.start_sec - prev_end < min_sec and kept[-1].kind == "cut" and t.kind == "cut":
            kept.pop()
        kept.append(t)
    transitions = [t.model_copy(update={"id": i}) for i, t in enumerate(kept, 1)]

    segments, cursor, prev_tid = [], 0.0, None
    for t in transitions:
        if t.start_sec - cursor > 1e-6:
            segments.append(Segment(id=len(segments) + 1, start_sec=cursor, end_sec=t.start_sec,
                                    transition_in=prev_tid, transition_out=t.id))
        cursor, prev_tid = t.end_sec, t.id
    if duration - cursor > 1e-6:
        segments.append(Segment(id=len(segments) + 1, start_sec=cursor, end_sec=round(duration, 3),
                                transition_in=prev_tid, transition_out=None))
    return transitions, segments
