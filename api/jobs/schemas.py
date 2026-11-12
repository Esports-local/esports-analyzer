"""/jobs 요청·응답 모양 (api ↔ analyzer 연동 스펙, docs/PLAN.md 6장). api 는 이 모양만 믿는다."""

from typing import Literal

from pydantic import BaseModel

JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


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
