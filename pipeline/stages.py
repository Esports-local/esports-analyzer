"""파이프라인 단계 정의와 실행 함수.

코드에서 바로 실행한다 (CLI 는 이 함수를 감싼 것뿐이고, 나중에 워커도 같은 함수를 부른다).

    from pipeline import run_pipeline

    run_pipeline("/data/ep01.mp4")                                   # 구현된 전 단계
    run_pipeline("/data/ep01.mp4", stages=["segments", "faces"])     # 고른 단계만 (순서는 STAGES 순)
    run_pipeline("/data/ep01.mp4", start="segments", until="faces")  # 구간 (미구현 단계는 건너뜀)
    run_pipeline("/data/ep01.mp4", stages=["faces"], params={"faces": {"cluster_threshold": 0.5}})

각 단계는 work_dir/work/<stage>.json 을 출력한다 (hls 는 hls/index.m3u8).
"""

import importlib
import logging
from pathlib import Path

log = logging.getLogger(__name__)

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

# 구현된 단계 → 모듈. 모듈마다 run(video, work_dir, params) 를 가진다
IMPLEMENTED = {
    "preprocess": "pipeline.preprocess",
    "segments": "pipeline.segments",
    "hls": "pipeline.hls",
    "faces": "pipeline.faces",
}


def run_stage(name: str, video: str | Path, work_dir: str | Path, params: dict | None = None) -> Path:
    """단계 하나를 실행한다.

    Args:
        name: 단계 이름 (STAGES 중 하나).
        video: 원본 영상 경로.
        work_dir: 회차 작업 디렉터리 (outputs/<영상명>/).
        params: 그 단계의 DEFAULT_PARAMS 를 덮어쓸 설정.

    Returns:
        저장된 단계 결과 파일 경로 (work/<stage>.json, hls 는 hls/index.m3u8).

    Raises:
        ValueError: 알 수 없는 단계 이름.
        NotImplementedError: 아직 구현하지 않은 단계.
    """
    if name not in STAGES:
        raise ValueError(f"unknown stage: {name}")
    if name not in IMPLEMENTED:
        raise NotImplementedError(f"stage '{name}' is not implemented yet")
    module = importlib.import_module(IMPLEMENTED[name])
    return module.run(Path(video), Path(work_dir), params)


def run_pipeline(
    video: str | Path,
    out_root: str | Path = "outputs",
    stages: list[str] | None = None,
    start: str | None = None,
    until: str | None = None,
    params: dict[str, dict] | None = None,
) -> dict[str, Path | None]:
    """영상 하나에 대해 여러 단계를 STAGES 순서대로 실행한다.

    아직 구현하지 않은 단계는 경고만 남기고 건너뛴다. 단계끼리는 work/ 의 중간 결과로 이어지므로,
    앞 단계를 이미 돌렸다면 뒤 단계만 골라 실행해도 된다 (예: faces 는 work/segments.json 이 필요).

    Args:
        video: 원본 영상 경로.
        out_root: 출력 루트. 결과는 <out_root>/<영상 파일명>/ 아래에 저장된다.
        stages: 실행할 단계 목록. 주면 start/until 은 쓸 수 없다. 순서는 STAGES 순으로 맞춘다.
        start: 시작 단계 (포함). 없으면 처음부터.
        until: 끝 단계 (포함). 없으면 끝까지.
        params: {단계 이름: 설정} 단계별 설정 덮어쓰기.

    Returns:
        {단계 이름: 결과 파일 경로}. 건너뛴 단계는 None.

    Raises:
        ValueError: 알 수 없는 단계, start 가 until 보다 뒤, stages 와 start/until 을 같이 줌.
    """
    if stages is not None:
        if start or until:
            raise ValueError("stages 와 start/until 은 같이 쓸 수 없습니다")
        unknown = [s for s in stages if s not in STAGES]
        if unknown:
            raise ValueError(f"unknown stage: {unknown}")
        selected = [s for s in STAGES if s in stages]
    else:
        for s in (start, until):
            if s is not None and s not in STAGES:
                raise ValueError(f"unknown stage: {s}")
        i, j = STAGES.index(start or STAGES[0]), STAGES.index(until or STAGES[-1])
        if i > j:
            raise ValueError(f"start({start}) 가 until({until}) 보다 뒤 단계입니다")
        selected = STAGES[i : j + 1]

    work_dir = Path(out_root) / Path(video).stem
    work_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path | None] = {}
    for name in selected:
        if name not in IMPLEMENTED:
            log.warning("[%s] 아직 구현되지 않아 건너뜀", name)
            results[name] = None
            continue
        log.info("[%s] 시작", name)
        results[name] = run_stage(name, video, work_dir, (params or {}).get(name))
        log.info("[%s] 완료 → %s", name, results[name])
    return results
