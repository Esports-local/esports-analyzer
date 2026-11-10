"""단계 결과 저장/읽기. 경로: <work_dir>/work/<stage>.json (중간 결과)"""

from pathlib import Path

from pydantic import BaseModel

from pipeline.schema import StageOutput


def stage_path(work_dir: Path, stage: str) -> Path:
    return work_dir / "work" / f"{stage}.json"


def save_stage(work_dir: Path, stage: str, result: BaseModel, params: dict) -> Path:
    out = StageOutput[type(result)](stage=stage, episode=work_dir.name, params=params, result=result)
    path = stage_path(work_dir, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(out.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_stage(work_dir: Path, stage: str, result_type: type[BaseModel]) -> StageOutput:
    return StageOutput[result_type].model_validate_json(stage_path(work_dir, stage).read_text(encoding="utf-8"))
