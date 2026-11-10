"""segments 단계: TransNetV2 로 전환을 찾고, 전환 사이를 segment 로 만든다.

1. 전체 프레임을 48x27 로 읽어 TransNetV2 에 넣는다.
   - single: 전환 지점 확률 (컷이면 컷 직전 프레임 하나가 높다)
   - many:   전환에 속한 모든 프레임 확률 (점진 전환이면 구간 전체가 높다)
2. 전환 후보를 찾는다.
   - 점진 전환: many > many_threshold 가 gradual_min_frames 이상 이어지는 구간.
     양끝은 many > many_low 인 곳까지 넓힌다 (확률이 서서히 오르내리므로)
   - cut: single > single_threshold 인 지점 중 점진 전환 구간 밖에 있는 것 (한 프레임)
   - 사이가 아주 짧거나(min_segment_sec 미만) 거의 검은 화면인 전환들은 하나로 합친다
     (페이드 아웃 → 검은 화면 → 페이드 인 이 한 전환이 되도록)
3. 종류는 cut / gradual 만 구분한다 (디졸브·와이프 구분은 하지 않음).
   구간 안에 거의 검은 프레임이 있으면 black=True (암전).
4. cut 두 개 사이가 min_segment_sec 미만이면(플래시 등) 앞 cut 을 지워 앞 segment 에 합친다.

시간: 프레임 i 는 [i/fps, (i+1)/fps) 구간. 프레임은 fps 필터로 CFR 에 맞춰 읽으므로 VFR 영상도 시각이 맞다.
"""

from pathlib import Path

import numpy as np

from pipeline.io import save_stage
from pipeline.media import fps_of, probe, read_frames, video_stream
from pipeline.schema import Segment, SegmentsResult, Transition

DEFAULT_PARAMS = {
    "detector": "transnetv2",
    "single_threshold": 0.5,
    "many_threshold": 0.5,
    "many_low": 0.2,
    "gradual_min_frames": 3,
    "fade_black_luma": 20.0,
    "min_segment_sec": 0.3,
}


def run(video: Path, work_dir: Path, params: dict | None = None) -> Path:
    """segments 단계를 실행한다.

    프레임을 48x27, CFR 로 읽어 TransNetV2 로 전환을 찾고, 전환 사이를 segment 로 만든다.

    Args:
        video: 원본 영상 경로.
        work_dir: 회차 작업 디렉터리.
        params: DEFAULT_PARAMS 를 덮어쓸 설정 (임계값 등). 실제 쓴 값과 fps 가 결과에 기록된다.

    Returns:
        work/segments.json 경로.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    fps = fps_of(video_stream(probe(video)))
    frames = read_frames(video, 48, 27, fps)
    single, many = predict(frames)
    luma = frames.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    luma = luma.mean(axis=(1, 2))
    spans = find_transition_spans(single, many, luma, fps, p)
    transitions = [classify(luma, a, b, gradual, i, fps, p) for i, (a, b, gradual) in enumerate(spans, 1)]
    duration = len(frames) / fps
    transitions, segments = build_segments(transitions, duration, p["min_segment_sec"])
    p["fps"] = fps
    return save_stage(work_dir, "segments", SegmentsResult(transitions=transitions, segments=segments), p)


def predict(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """TransNetV2 로 프레임마다 전환 확률을 계산한다 (CPU).

    Args:
        frames: (N, 27, 48, 3) uint8 RGB. TransNetV2 는 이 크기만 받는다.

    Returns:
        (single, many). 둘 다 길이 N 의 0~1 확률.
        single 은 컷 지점(컷 직전 프레임), many 는 전환 구간에 속한 프레임일 확률.
    """
    import torch
    from transnetv2_pytorch import TransNetV2

    model = TransNetV2(device="cpu")
    model.eval()
    with torch.no_grad():
        single, many = model.predict_frames(torch.from_numpy(frames.copy()), quiet=True)
    return single.numpy(), many.numpy()


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """bool 배열에서 True 가 연속된 구간을 찾는다.

    Args:
        mask: bool 배열.

    Returns:
        (a, b) 목록. a, b 는 양끝을 포함한 인덱스.
    """
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
    """전환 확률에서 전환 구간을 찾는다.

    - many > many_threshold 가 gradual_min_frames 이상 이어지면 점진 전환. 양끝은 many > many_low 까지 넓힌다.
    - single > single_threshold 인 지점 중 점진 전환 밖에 있으면 컷 (한 프레임).
    - 겹치거나, 사이가 검은 화면이거나, 사이가 min_segment_sec 미만인데 점진 전환이 끼어 있으면 하나로 합친다.

    Args:
        single: 컷 지점 확률 (predict() 결과).
        many: 전환 구간 확률 (predict() 결과).
        luma: 프레임별 평균 밝기 (0~255).
        fps: 프레임레이트.
        p: 설정 (DEFAULT_PARAMS 형식).

    Returns:
        (a, b, gradual) 목록. a, b 는 양끝을 포함한 프레임 번호.
        컷이면 a == b (컷 직전 프레임) 이고 gradual 은 False.
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

    # 겹치거나, 사이가 짧은데 점진 전환이 끼어 있거나, 사이가 검은 화면이면 합친다
    min_gap = p["min_segment_sec"] * fps
    merged: list[tuple[int, int, bool]] = []
    for a, b, g in spans:
        if merged:
            pa, pb, pg = merged[-1]
            gap = a - pb - 1
            black_gap = gap > 0 and luma[pb + 1 : a].max() < p["fade_black_luma"]
            if gap <= 0 or black_gap or (gap < min_gap and (g or pg)):
                merged[-1] = (pa, max(pb, b), True)
                continue
        merged.append((a, b, g))
    return merged


def classify(
    luma: np.ndarray, a: int, b: int, gradual: bool, tid: int, fps: float, p: dict
) -> Transition:
    """전환 구간 하나를 Transition 으로 만든다.

    구간 안에 밝기 < fade_black_luma 인 프레임이 있으면 black=True (암전).

    Args:
        luma: 프레임별 평균 밝기.
        a: 구간 시작 프레임.
        b: 구간 끝 프레임 (포함).
        gradual: 점진 전환이면 True, 컷이면 False.
        tid: 전환 id.
        fps: 프레임레이트.
        p: 설정.

    Returns:
        컷은 start_sec == end_sec == (a + 1) / fps (컷 뒤 첫 프레임의 시작 시각).
        점진 전환은 [a / fps, (b + 1) / fps].
    """
    black = bool(luma[a : b + 1].min() < p["fade_black_luma"])
    if not gradual:
        t = round((a + 1) / fps, 3)
        return Transition(id=tid, kind="cut", start_sec=t, end_sec=t, black=black)
    start, end = round(a / fps, 3), round((b + 1) / fps, 3)
    return Transition(id=tid, kind="gradual", start_sec=start, end_sec=end, black=black)


def build_segments(
    transitions: list[Transition], duration: float, min_sec: float
) -> tuple[list[Transition], list[Segment]]:
    # 너무 짧은 segment 는 그 앞 전환을 지워 앞 segment 에 합친다
    """전환 목록으로 segment 목록을 만든다.

    컷 두 개 사이가 min_sec 미만이면(플래시 등) 앞 컷을 지워 앞 segment 에 합친다.
    전환 구간 자체는 어떤 segment 에도 넣지 않는다.

    Args:
        transitions: classify() 결과 목록 (시간순).
        duration: 영상 길이(초).
        min_sec: segment 최소 길이(초).

    Returns:
        (transitions, segments). 지운 전환이 있으면 전환 id 를 1부터 다시 매긴다.
    """
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
