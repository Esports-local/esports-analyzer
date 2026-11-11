"""scenes 단계: 인접 segment 를 같은 장면이면 묶어 scene 을 만든다 (초기 규칙, 기준값은 params 로 조정).

경계마다 판단 (앞 segment → 뒤 segment)
1. 암전(점진 전환 + black)이면 무조건 나눈다 (split_on_black). 시간 경과·큰 장면 전환 신호.
2. 대사가 경계를 넘으면(dialogue = 1.0) 묶는다. 하나의 발화가 컷을 넘어가면 같은 장면.
3. 하드컷: visual ≥ visual_threshold 또는 (같은 화자가 이어 말함(0.5) 이고 visual ≥ visual_threshold_with_speaker) 이면 묶는다.
4. 점진 전환(디졸브·와이프): visual ≥ gradual_visual_threshold 일 때만 묶는다 (보통 장면 전환이라 더 엄격하게).
5. 그 외는 나눈다.

후처리
- min_scene_sec 보다 짧은 scene 은 앞뒤 중 경계 visual 이 더 높은 쪽에 합친다 (암전 경계 너머로는 합치지 않음).

scene 시간 범위는 첫 segment 시작 ~ 마지막 segment 끝. 씬 안의 점진 전환 구간도 포함된다.

입력: work/segments.json, work/similarity.json
출력: work/scenes.json
"""

from pathlib import Path

from pipeline.io import load_stage, save_stage
from pipeline.schema import Scene, ScenesResult, SegmentPair, SegmentsResult, SimilarityResult, Transition

DEFAULT_PARAMS = {
    "split_on_black": True,
    "visual_threshold": 0.80,
    "visual_threshold_with_speaker": 0.65,
    "gradual_visual_threshold": 0.92,
    "min_scene_sec": 4.0,
}


def run(video: Path, work_dir: Path, params: dict | None = None) -> Path:
    """scenes 단계를 실행한다.

    Args:
        video: 원본 영상 경로 (쓰지 않음).
        work_dir: 회차 작업 디렉터리. work/segments.json, work/similarity.json 이 있어야 한다.
        params: DEFAULT_PARAMS 를 덮어쓸 설정.

    Returns:
        work/scenes.json 경로.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    seg = load_stage(work_dir, "segments", SegmentsResult).result
    pairs = load_stage(work_dir, "similarity", SimilarityResult).result.pairs
    scenes = build_scenes(seg, pairs, p)
    return save_stage(work_dir, "scenes", ScenesResult(scenes=scenes), p)


def same_scene(pair: SegmentPair | None, transition: Transition | None, p: dict) -> bool:
    """경계 하나에서 앞뒤 segment 를 같은 장면으로 볼지 판단한다.

    Args:
        pair: similarity 결과 (없으면 나눈다).
        transition: 두 segment 사이 전환 (없으면 하드컷으로 본다).
        p: 설정.

    Returns:
        묶으면 True.
    """
    if pair is None:
        return False
    gradual = transition is not None and transition.kind == "gradual"
    if gradual and transition.black and p["split_on_black"]:
        return False
    if pair.dialogue == 1.0:
        return True
    if gradual:
        return pair.visual >= p["gradual_visual_threshold"]
    if pair.visual >= p["visual_threshold"]:
        return True
    return pair.dialogue == 0.5 and pair.visual >= p["visual_threshold_with_speaker"]


def build_scenes(seg: SegmentsResult, pairs: list[SegmentPair], p: dict) -> list[Scene]:
    """segment 와 경계 점수로 scene 목록을 만든다.

    Args:
        seg: segments 단계 결과.
        pairs: similarity 단계 결과 (인접 segment 쌍).
        p: 설정.

    Returns:
        시간순 Scene 목록 (id 는 1부터).
    """
    segments = seg.segments
    if not segments:
        return []
    by_pair = {(x.prev_id, x.next_id): x for x in pairs}
    tr = {t.id: t for t in seg.transitions}

    def boundary(i: int) -> tuple[SegmentPair | None, Transition | None]:
        a, b = segments[i], segments[i + 1]
        return by_pair.get((a.id, b.id)), tr.get(a.transition_out)

    # 1차: 경계마다 판단해 묶음 (segment 인덱스 목록)
    groups: list[list[int]] = [[0]]
    for i in range(len(segments) - 1):
        pair, t = boundary(i)
        if same_scene(pair, t, p):
            groups[-1].append(i + 1)
        else:
            groups.append([i + 1])

    def is_black(i: int) -> bool:
        _, t = boundary(i)
        return t is not None and t.kind == "gradual" and t.black

    def visual(i: int) -> float:
        pair, _ = boundary(i)
        return pair.visual if pair else -1.0

    # 2차: 짧은 scene 을 이웃에 합친다
    changed = True
    while changed and len(groups) > 1:
        changed = False
        for gi, g in enumerate(groups):
            dur = segments[g[-1]].end_sec - segments[g[0]].start_sec
            if dur >= p["min_scene_sec"]:
                continue
            left_ok = gi > 0 and not (p["split_on_black"] and is_black(g[0] - 1))
            right_ok = gi < len(groups) - 1 and not (p["split_on_black"] and is_black(g[-1]))
            if not left_ok and not right_ok:
                continue
            left_v = visual(g[0] - 1) if left_ok else -2.0
            right_v = visual(g[-1]) if right_ok else -2.0
            if left_v >= right_v:
                groups[gi - 1].extend(g)
            else:
                groups[gi + 1][:0] = g
            del groups[gi]
            changed = True
            break

    scenes = []
    for g in groups:
        first, last = segments[g[0]], segments[g[-1]]
        before = tr.get(first.transition_in)
        scenes.append(Scene(
            id=len(scenes) + 1, start_sec=first.start_sec, end_sec=last.end_sec,
            segment_ids=[segments[i].id for i in g],
            black_before=bool(before is not None and before.kind == "gradual" and before.black),
        ))
    return scenes
