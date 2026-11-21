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


def skip_spans(work_dir: Path, kinds: list[str], in_game: list[str] | None = None) -> list[tuple[float, float]]:
    """segments 결과에서 kind 가 kinds 중 하나인 구간 [(start, end)] (원본 영상 초). segments 가 없으면 빈 목록.

    roi_ocr · asr 가 필요 없는 화면(밴픽·쉬는 시간·카운트다운 등)을 건너뛸 때 쓴다.
    경기 구간(segments.game) 안에서는 in_game 에 있는 종류만 건너뛴다 (경기 중 캠·배너 장면에도 해설·점수판이 있다).
    """
    import json

    path = stage_path(work_dir, "segments")
    if not kinds or not path.exists():
        return []
    result = json.loads(path.read_text(encoding="utf-8")).get("result") or {}
    game = result.get("game") or {}
    g0, g1 = game.get("start_sec"), game.get("end_sec")
    in_game = set(in_game or [])
    spans: list[tuple[float, float]] = []
    for s in result.get("segments", []):
        inside = g0 is not None and g0 <= s["start_sec"] < (g1 if g1 is not None else float("inf"))
        if s.get("kind") in kinds and (not inside or s["kind"] in in_game):
            if spans and abs(spans[-1][1] - s["start_sec"]) < 1.0:  # 붙어 있으면 (전환 구간 포함) 하나로
                spans[-1] = (spans[-1][0], s["end_sec"])
            else:
                spans.append((s["start_sec"], s["end_sec"]))
    return spans


def in_spans(spans: list[tuple[float, float]], t: float) -> bool:
    return any(a <= t < b for a, b in spans)

