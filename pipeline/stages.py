"""파이프라인 단계 정의. 각 단계는 work_dir/work/<stage>.json 을 출력한다."""

from pathlib import Path

STAGES = [
    "preprocess",  # ffprobe 메타데이터·VFR 점검 + 16kHz 모노 wav
    "segments",    # 전환 검출 (cut/gradual + 암전) -> segment
    "hls",         # 전환 경계 기준 ts(<=4초) 재인코딩
    "keyframes",   # segment별 키프레임 추출
    "faces",       # InsightFace 얼굴 검출·임베딩 -> 회차 내 군집 F1, F2 …
    "asr",         # Silero VAD + faster-whisper (word timestamps)
    "similarity",  # 인접 segment 연속성 (MobileCLIP 등)
    "scenes",      # segment -> scene 병합
    "vlm",         # scene별 대표 프레임 + 대사 -> VLM
    "select",      # 중요 scene 선별
    "predict",     # 다음 화 예측 + 근거
]


def run_stage(name: str, video: Path, work_dir: Path) -> Path:
    if name not in STAGES:
        raise ValueError(f"unknown stage: {name}")
    if name == "preprocess":
        from pipeline import preprocess

        return preprocess.run(video, work_dir)
    if name == "segments":
        from pipeline import segments

        return segments.run(video, work_dir)
    if name == "faces":
        from pipeline import faces

        return faces.run(video, work_dir)
    raise NotImplementedError(f"stage '{name}' is not implemented yet")
