"""파이프라인 단계 정의. 각 단계는 work_dir/<stage>.json 을 출력한다."""

from pathlib import Path

STAGES = [
    "preprocess",  # 영상/음성 분리, 정규화
    "shots",       # PySceneDetect 하드컷 검출
    "keyframes",   # Shot별 키프레임 추출
    "asr",         # Silero VAD + faster-whisper (word timestamps)
    "similarity",  # MobileCLIP 화면 유사도
    "merge",       # Shot -> Segment 병합
    "vlm",         # Segment별 대표 프레임 + 대사 -> VLM
    "select",      # 중요 Segment 선별
    "predict",     # 다음 화 예측 + 근거
]


def run_stage(name: str, video: Path, work_dir: Path) -> Path:
    if name not in STAGES:
        raise ValueError(f"unknown stage: {name}")
    raise NotImplementedError(f"stage '{name}' is not implemented yet")
