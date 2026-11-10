"""segments 단계: TransNetV2 로 전환을 찾고, 전환 사이를 segment 로 만든다.

1. 전체 프레임을 48x27 로 읽어 TransNetV2 에 넣는다.
   - single: 전환 지점 확률 (컷이면 컷 직전 프레임 하나가 높다)
   - many:   전환에 속한 모든 프레임 확률 (점진 전환이면 구간 전체가 높다)
2. 전환 후보를 찾는다.
   - 점진 전환: many > many_threshold 가 gradual_min_frames 이상 이어지는 구간.
     양끝은 many > many_low 인 곳까지 넓힌다 (확률이 서서히 오르내리므로)
   - cut: single > single_threshold 인 지점 중 점진 전환 구간 밖에 있는 것 (한 프레임)
   - 사이가 아주 짧거나(min_segment_sec 미만), 짧은(max_black_gap_sec 이하) 완전한 검정이면 하나로 합친다
     (페이드 아웃 → 검은 화면 → 페이드 인 이 한 전환이 되도록. 긴 밤 장면은 합치지 않음)
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
    "fade_black_luma": 20.0,  # 전환 구간 안에 이보다 어두운 프레임이 있으면 black=True
    "black_gap_luma": 10.0,  # 두 전환 사이가 이보다 어두우면 (거의 완전한 검정) 하나로 합칠 후보
    "max_black_gap_sec": 2.0,  # 그 검정 구간이 이 길이 이하일 때만 합친다 (밤 장면 보호)
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
    luma = frame_luma(frames)
    single, many = predict(frames)
    duration = len(frames) / fps
    del frames
    spans = find_transition_spans(single, many, luma, fps, p)
    transitions = [classify(luma, a, b, gradual, i, fps, p) for i, (a, b, gradual) in enumerate(spans, 1)]
    transitions, segments = build_segments(transitions, duration, p["min_segment_sec"])
    p["fps"] = fps
    return save_stage(work_dir, "segments", SegmentsResult(transitions=transitions, segments=segments), p)


def frame_luma(frames: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """프레임별 평균 밝기(0~255)를 계산한다.

    전체를 한 번에 float 로 바꾸면 프레임 배열의 4배 메모리가 들어서, chunk 프레임씩 나눠 계산한다.

    Args:
        frames: (N, H, W, 3) uint8 RGB.
        chunk: 한 번에 계산할 프레임 수.

    Returns:
        (N,) float32 평균 밝기.
    """
    weights = np.array([0.299, 0.587, 0.114], np.float32)
    out = np.empty(len(frames), np.float32)
    for i in range(0, len(frames), chunk):
        part = frames[i : i + chunk]
        out[i : i + len(part)] = (part.reshape(len(part), -1, 3) @ weights).mean(axis=1)
    return out


def predict(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """TransNetV2 로 프레임마다 전환 확률을 계산한다 (CPU).

    Args:
        frames: (N, 27, 48, 3) uint8 RGB. TransNetV2 는 이 크기만 받는다.
            쓰기 가능한 배열이어야 복사 없이 torch 로 넘어간다 (read_frames() 결과는 쓰기 가능).

    Returns:
        (single, many). 둘 다 길이 N 의 0~1 확률.
        single 은 컷 지점(컷 직전 프레임), many 는 전환 구간에 속한 프레임일 확률.
    """
    import torch
    from transnetv2_pytorch import TransNetV2

    model = TransNetV2(device="cpu")
    model.eval()
    with torch.no_grad():
        single, many = model.predict_frames(torch.from_numpy(frames), quiet=True)
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
    - 다음 경우 앞뒤 전환을 하나로 합친다.
      · 구간이 겹침
      · 사이가 min_segment_sec 미만인데 점진 전환이 끼어 있음
      · 사이가 거의 완전한 검정(밝기 < black_gap_luma)이고 max_black_gap_sec 이하 (페이드 아웃 → 검정 → 페이드 인).
        길이 제한이 없으면 어두운 밤 장면 전체가 전환으로 사라지므로 짧은 검정만 합친다.

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


def classify(
    luma: np.ndarray, a: int, b: int, gradual: bool, tid: int, fps: float, p: dict
) -> Transition:
    """전환 구간 하나를 Transition 으로 만든다.

    점진 전환 구간 안에 밝기 < fade_black_luma 인 프레임이 있으면 black=True (암전). 컷은 항상 black=False.

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
    if not gradual:
        # 컷은 한 시점이라 암전이 아니다 (어두운 장면에서 나오는 컷도 black 이 아님).
        # 컷 → 짧은 검정 → 컷 은 find_transition_spans 에서 이미 점진 전환 하나로 합쳐진다.
        t = round((a + 1) / fps, 3)
        return Transition(id=tid, kind="cut", start_sec=t, end_sec=t, black=False)
    black = bool(luma[a : b + 1].min() < p["fade_black_luma"])
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
