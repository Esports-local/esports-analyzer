"""vlm 단계: scene 마다 대표 프레임과 대사를 VLM 에 넣어 지문(summary)·유형(type)·키워드를 만든다.

백엔드 (params: backend)
- "mock": 로컬 개발용. 모델 없이 대사로 그럴듯한 값을 만든다 (형식 확인·api/ui 샘플용)
- "transformers": GPU 서버에서 Qwen3-VL. 아직 구현 안 함 (자리만)

대표 프레임은 scene 안의 segment 들에서 고르게 frames_per_scene 장을 고른다 (전환 구간 제외).
프레임이 필요 없는 백엔드(mock)는 프레임을 읽지 않는다.

입력: work/scenes.json, work/segments.json, work/asr.json (있으면)
출력: work/vlm.json
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from pipeline.io import load_stage, save_stage
from pipeline.media import frame_size, read_frame_at
from pipeline.schema import AsrResult, Line, Scene, SceneAnalysis, SegmentsResult, ScenesResult, VlmResult

DEFAULT_PARAMS = {
    "backend": "mock",
    "model": None,  # transformers 백엔드: 예) Qwen/Qwen3-VL-4B-Instruct
    "frames_per_scene": 4,
    "frame_height": 448,
    "max_dialogue_chars": 2000,  # 프롬프트에 넣을 대사 최대 길이
}

SCENE_TYPES = ["talk", "action", "reaction", "transition", "montage", "mixed"]


def run(video: Path, work_dir: Path, params: dict | None = None) -> Path:
    """vlm 단계를 실행한다.

    Args:
        video: 원본 영상 경로.
        work_dir: 회차 작업 디렉터리. work/scenes.json, work/segments.json 이 있어야 한다.
        params: DEFAULT_PARAMS 를 덮어쓸 설정.

    Returns:
        work/vlm.json 경로.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    scenes = load_stage(work_dir, "scenes", ScenesResult).result.scenes
    segments = {s.id: s for s in load_stage(work_dir, "segments", SegmentsResult).result.segments}
    try:
        lines = load_stage(work_dir, "asr", AsrResult).result.lines
    except FileNotFoundError:
        lines = []

    backend = make_backend(p)
    size = frame_size(video, p["frame_height"]) if backend.needs_frames else None
    results = []
    for scene in scenes:
        times = pick_frame_times(scene, segments, p["frames_per_scene"])
        frames = [f for t in times if (f := read_frame_at(video, t, *size)) is not None] if size else []
        scene_lines = lines_in(scene, lines)
        out = backend.describe(frames, scene_lines, p)
        results.append(SceneAnalysis(scene_id=scene.id, type=out["type"], summary=out["summary"],
                                     keywords=out.get("keywords", []), frame_times=[round(t, 3) for t in times]))
    return save_stage(work_dir, "vlm", VlmResult(backend=backend.name, model=p["model"], scenes=results), p)


def pick_frame_times(scene: Scene, segments: dict, n: int) -> list[float]:
    """scene 안의 segment 들에서 고르게 n 개 시각을 고른다 (전환 구간은 segment 밖이라 자연히 제외).

    segment 를 시간순으로 이어 붙인 "내용 시간" 위에서 등간격으로 고른 뒤 실제 시각으로 바꾼다.

    Args:
        scene: 대상 scene.
        segments: {segment id: Segment}.
        n: 고를 개수.

    Returns:
        시간순 시각(초) 목록.
    """
    segs = [segments[i] for i in scene.segment_ids if i in segments]
    total = sum(s.end_sec - s.start_sec for s in segs)
    if not segs or total <= 0 or n <= 0:
        return []
    times = []
    for k in range(n):
        target = total * (k + 0.5) / n
        for s in segs:
            d = s.end_sec - s.start_sec
            if target <= d:
                times.append(s.start_sec + target)
                break
            target -= d
    return times


def lines_in(scene: Scene, lines: list[Line]) -> list[Line]:
    """시작 시각이 scene 안에 있는 대사."""
    return [ln for ln in lines if scene.start_sec <= ln.start_sec < scene.end_sec]


def build_prompt(lines: list[Line], max_chars: int) -> str:
    """VLM 프롬프트를 만든다 (실제 VLM 백엔드용).

    Args:
        lines: scene 의 대사.
        max_chars: 대사 부분 최대 길이 (넘으면 뒤를 자른다).

    Returns:
        프롬프트 문자열. 모델은 JSON {"type", "summary", "keywords"} 로 답해야 한다.
    """
    dialogue = "\n".join(f"{ln.speaker or '?'}: {ln.text}" for ln in lines)
    if len(dialogue) > max_chars:
        dialogue = dialogue[:max_chars] + "\n…"
    return (
        "드라마의 한 장면을 보여 준다. 이미지는 장면 안의 대표 프레임을 시간순으로 고른 것이다.\n"
        "화면에 보이는 상황을 대본의 지문처럼 한두 문장으로 쓰고, 장면 유형과 키워드를 고른다.\n"
        "인물 이름은 모르므로 '여자', '남자', '두 사람'처럼 쓴다. 장소 이름은 쓰지 않는다.\n"
        f"유형: {', '.join(SCENE_TYPES)} 중 하나\n"
        "  talk=대화 중심, action=행동·사건, reaction=표정·반응, transition=시간 변화·회상 진입,"
        " montage=짧은 장면 연결, mixed=혼합\n"
        f"대사 (화자 라벨: 대사):\n{dialogue or '(대사 없음)'}\n\n"
        '다음 JSON 으로만 답한다: {"type": "...", "summary": "...", "keywords": ["...", "..."]}'
    )


def parse_answer(text: str) -> dict:
    """VLM 답에서 JSON 을 꺼내 검증한다. 형식이 틀리면 mixed / 빈 지문으로 둔다."""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        data = {}
    t = data.get("type") if data.get("type") in SCENE_TYPES else "mixed"
    kw = [str(k) for k in data.get("keywords", []) if str(k).strip()][:8]
    return {"type": t, "summary": str(data.get("summary", "")).strip(), "keywords": kw}


class MockBackend:
    """모델 없이 대사로 값을 만드는 개발용 백엔드."""

    name = "mock"
    needs_frames = False

    def describe(self, frames: list[np.ndarray], lines: list[Line], p: dict) -> dict:
        """대사 유무·화자로 유형과 지문을 만들고, 자주 나온 단어를 키워드로 쓴다.

        Args:
            frames: 쓰지 않음.
            lines: scene 의 대사.
            p: 설정.

        Returns:
            {"type", "summary", "keywords"}.
        """
        if not lines:
            return {"type": "action", "summary": "(mock) 대사 없는 장면", "keywords": []}
        speakers = sorted({ln.speaker for ln in lines if ln.speaker})
        who = ", ".join(speakers) if speakers else "화자 미상"
        words = [w for ln in lines for w in re.findall(r"[가-힣A-Za-z0-9ぁ-んァ-ン一-龥]{2,}", ln.text)]
        keywords = [w for w, _ in Counter(words).most_common(5)]
        return {"type": "talk", "summary": f"(mock) {who}의 대화, 대사 {len(lines)}줄", "keywords": keywords}


class TransformersBackend:
    """GPU 서버용 Qwen3-VL 백엔드 (아직 구현 안 함)."""

    name = "transformers"
    needs_frames = True

    def __init__(self, model: str | None):
        """모델 이름을 받는다. 실제 로딩은 구현 시 추가."""
        self.model = model

    def describe(self, frames: list[np.ndarray], lines: list[Line], p: dict) -> dict:
        """build_prompt() 와 frames 로 모델을 호출하고 parse_answer() 로 결과를 꺼낼 예정."""
        raise NotImplementedError("transformers VLM 백엔드는 GPU 서버 작업 때 구현한다")


def make_backend(p: dict):
    """설정으로 백엔드를 만든다.

    Raises:
        ValueError: 알 수 없는 backend.
    """
    if p["backend"] == "mock":
        return MockBackend()
    if p["backend"] == "transformers":
        return TransformersBackend(p["model"])
    raise ValueError(f"unknown vlm backend: {p['backend']}")
