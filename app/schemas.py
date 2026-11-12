"""api ↔ analyzer 연동 스펙 (docs/PLAN.md 6장). api 는 이 모양만 믿는다."""

from typing import Literal

from pydantic import BaseModel

JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


class VideoInfo(BaseModel):
    name: str                # VIDEOS_ROOT 기준 파일 이름
    size_bytes: int
    duration_sec: float | None
    width: int | None
    height: int | None
    has_audio: bool
    selectable: bool         # 1920x1080 이고 소리가 있으면 true
    reason: str | None = None  # selectable 이 false 인 이유


class JobCreate(BaseModel):
    job_id: str
    video: str               # VIDEOS_ROOT 기준 파일 이름
    stages: list[str]
    params: dict[str, dict] = {}  # {단계: 설정} (pipeline --set 으로 넘김)


class JobState(BaseModel):
    job_id: str
    video: str
    work_name: str           # outputs/<work_name>/ (영상 파일 이름에서 확장자 뺀 것)
    stages: list[str]
    status: JobStatus
    current_stage: str | None = None
    elapsed_sec: float = 0.0
    error: str | None = None


class EstimateRequest(BaseModel):
    video: str
    conditions: list[str] = ["A", "B"]


class EstimateResult(BaseModel):
    model: str
    lines: int
    requests: int
    input_tokens: dict[str, int]
    sampled_lines: int  # 실제로 센 대사 수 (나머지는 비율로 늘림)
