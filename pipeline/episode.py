"""episode 단계: 단계별 중간 결과를 모아 최종 결과물 analysis/episode.json (영상 내용) 을 만든다.

- scene: scenes + vlm (지문·유형·키워드) + 암전 여부
- 대사: asr 의 각 줄을 시작 시각이 속한 scene 에 넣는다. scene 사이(전환 구간)에서 시작한 줄은 앞 scene 에 넣는다.
- 얼굴: faces 의 라벨 있는 검출을 scene 별로 모은다 (많이 나온 순).
- 화자 요약 / 얼굴 요약: api 의 매핑 화면용.

asr·faces·vlm 이 없으면 그 부분만 비워서 만든다 (단계를 일부만 돌렸을 때도 형식 확인 가능).

입력: work/preprocess.json, work/scenes.json, work/vlm.json, work/asr.json, work/faces.json
출력: analysis/episode.json
"""

from collections import Counter, defaultdict
from pathlib import Path

from pipeline.io import load_stage
from pipeline.schema import (
    AnalysisLine,
    AnalysisScene,
    AsrResult,
    EpisodeAnalysis,
    FaceSummary,
    FacesResult,
    Line,
    PreprocessResult,
    Scene,
    ScenesResult,
    SpeakerSummary,
    VlmResult,
)

DEFAULT_PARAMS = {
    "speaker_samples": 3,  # 화자 요약에 넣을 대표 대사 수 (긴 대사 순)
}


def run(video: Path, work_dir: Path, params: dict | None = None) -> Path:
    """episode 단계를 실행한다.

    Args:
        video: 원본 영상 경로 (쓰지 않음).
        work_dir: 회차 작업 디렉터리. work/preprocess.json, work/scenes.json 이 있어야 한다.
        params: DEFAULT_PARAMS 를 덮어쓸 설정.

    Returns:
        analysis/episode.json 경로.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    duration = load_stage(work_dir, "preprocess", PreprocessResult).result.duration_sec
    scenes = load_stage(work_dir, "scenes", ScenesResult).result.scenes
    vlm = {a.scene_id: a for a in _optional(work_dir, "vlm", VlmResult, "scenes")}
    lines = _optional(work_dir, "asr", AsrResult, "lines")
    faces = _optional_obj(work_dir, "faces", FacesResult)

    by_scene_lines = assign_lines(scenes, lines)
    by_scene_faces = faces_per_scene(scenes, faces)
    out_scenes = []
    for s in scenes:
        a = vlm.get(s.id)
        out_scenes.append(AnalysisScene(
            id=s.id, start_sec=s.start_sec, end_sec=s.end_sec,
            type=a.type if a else "mixed", summary=a.summary if a else "", keywords=a.keywords if a else [],
            black_before=s.black_before, faces=by_scene_faces.get(s.id, []),
            lines=[AnalysisLine(start_sec=ln.start_sec, end_sec=ln.end_sec, speaker=ln.speaker, text=ln.text)
                   for ln in by_scene_lines.get(s.id, [])],
        ))

    result = EpisodeAnalysis(
        episode=work_dir.name, duration_sec=duration, scenes=out_scenes,
        speakers=speaker_summary(lines, p["speaker_samples"]),
        faces=face_summary(scenes, faces),
    )
    path = work_dir / "analysis" / "episode.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path


def _optional(work_dir: Path, stage: str, result_type, attr: str) -> list:
    """단계 결과의 목록 필드를 읽는다. 파일이 없으면 빈 목록."""
    try:
        return getattr(load_stage(work_dir, stage, result_type).result, attr)
    except FileNotFoundError:
        return []


def _optional_obj(work_dir: Path, stage: str, result_type):
    """단계 결과를 읽는다. 파일이 없으면 None."""
    try:
        return load_stage(work_dir, stage, result_type).result
    except FileNotFoundError:
        return None


def assign_lines(scenes: list[Scene], lines: list[Line]) -> dict[int, list[Line]]:
    """대사를 시작 시각이 속한 scene 에 넣는다.

    scene 사이(전환 구간)나 첫 scene 앞에서 시작한 대사는 시작 시각 직전의 scene (없으면 첫 scene) 에 넣는다.

    Args:
        scenes: 시간순 scene.
        lines: asr 대사 줄.

    Returns:
        {scene id: 시간순 대사}.
    """
    out: dict[int, list[Line]] = defaultdict(list)
    if not scenes:
        return out
    starts = [s.start_sec for s in scenes]
    for ln in sorted(lines, key=lambda x: x.start_sec):
        idx = max((i for i, st in enumerate(starts) if st <= ln.start_sec), default=0)
        out[scenes[idx].id].append(ln)
    return out


def faces_per_scene(scenes: list[Scene], faces: FacesResult | None) -> dict[int, list[str]]:
    """scene 별로 나온 얼굴 라벨을 많이 나온 순서로 모은다.

    Args:
        scenes: scene 목록.
        faces: faces 단계 결과 (없으면 None).

    Returns:
        {scene id: [F1, F3, …]}.
    """
    if faces is None:
        return {}
    seg_to_scene = {sid: s.id for s in scenes for sid in s.segment_ids}
    counts: dict[int, Counter] = defaultdict(Counter)
    for d in faces.detections:
        if d.face_id and d.segment_id in seg_to_scene:
            counts[seg_to_scene[d.segment_id]][d.face_id] += 1
    return {sid: [f for f, _ in c.most_common()] for sid, c in counts.items()}


def speaker_summary(lines: list[Line], n_samples: int) -> list[SpeakerSummary]:
    """화자별 대사 수·총 시간·대표 대사 (api 매핑 화면용).

    Args:
        lines: asr 대사 줄.
        n_samples: 대표 대사 수 (긴 대사 순).

    Returns:
        화자 라벨 순 SpeakerSummary 목록. 화자 없는 대사는 제외.
    """
    groups: dict[str, list[Line]] = defaultdict(list)
    for ln in lines:
        if ln.speaker:
            groups[ln.speaker].append(ln)

    def order(label: str):
        return (0, int(label[1:])) if label[1:].isdigit() else (1, label)

    out = []
    for spk in sorted(groups, key=order):
        g = groups[spk]
        samples = [ln.text for ln in sorted(g, key=lambda x: len(x.text), reverse=True)[:n_samples]]
        out.append(SpeakerSummary(speaker=spk, line_count=len(g),
                                  total_sec=round(sum(ln.end_sec - ln.start_sec for ln in g), 3), samples=samples))
    return out


def face_summary(scenes: list[Scene], faces: FacesResult | None) -> list[FaceSummary]:
    """얼굴 라벨별 요약 (api 매핑 화면용). 대표 얼굴의 시각·위치로 hls 영상에서 잘라 보여 줄 수 있다.

    Args:
        scenes: scene 목록.
        faces: faces 단계 결과 (없으면 None).

    Returns:
        faces 군집 순서(F1, F2 …)의 FaceSummary 목록.
    """
    if faces is None:
        return []
    seg_to_scene = {sid: s.id for s in scenes for sid in s.segment_ids}
    det = {d.id: d for d in faces.detections}
    out = []
    for c in faces.clusters:
        best = det[c.best_detection_id]
        scene_ids = sorted({seg_to_scene[sid] for sid in c.segment_ids if sid in seg_to_scene})
        out.append(FaceSummary(face_id=c.face_id, count=c.count, scene_ids=scene_ids,
                               sample_time_sec=best.time_sec, sample_bbox=best.bbox))
    return out
