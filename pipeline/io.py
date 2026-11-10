"""단계 결과 저장/읽기. 경로: <work_dir>/work/<stage>.json (중간 결과)"""

from pathlib import Path

from pydantic import BaseModel

from pipeline.schema import StageOutput


def stage_path(work_dir: Path, stage: str) -> Path:
    """단계 결과 파일 경로를 돌려준다.

    Args:
        work_dir: 회차 작업 디렉터리.
        stage: 단계 이름.

    Returns:
        <work_dir>/work/<stage>.json
    """
    return work_dir / "work" / f"{stage}.json"


def save_stage(work_dir: Path, stage: str, result: BaseModel, params: dict) -> Path:
    """단계 결과를 공통 껍데기(StageOutput)로 감싸 JSON 으로 저장한다.

    저장 전에 Pydantic 으로 스키마를 검증한다. episode 는 work_dir 이름(영상 파일명)을 쓴다.

    Args:
        work_dir: 회차 작업 디렉터리.
        stage: 단계 이름.
        result: 단계 결과 모델 (예: SegmentsResult).
        params: 이 결과를 만들 때 쓴 설정. 그대로 기록된다.

    Returns:
        저장한 파일 경로.
    """
    out = StageOutput[type(result)](stage=stage, episode=work_dir.name, params=params, result=result)
    path = stage_path(work_dir, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(out.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_stage(work_dir: Path, stage: str, result_type: type[BaseModel]) -> StageOutput:
    """저장된 단계 결과를 읽고 스키마로 검증한다.

    Args:
        work_dir: 회차 작업 디렉터리.
        stage: 단계 이름.
        result_type: result 부분의 모델 타입 (예: SegmentsResult).

    Returns:
        검증된 StageOutput. 단계 결과는 .result 로 꺼낸다.

    Raises:
        FileNotFoundError: 앞 단계를 아직 실행하지 않음.
        pydantic.ValidationError: 파일 형식이 스키마와 다름.
    """
    return StageOutput[result_type].model_validate_json(stage_path(work_dir, stage).read_text(encoding="utf-8"))
