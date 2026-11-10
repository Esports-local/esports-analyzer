from typing import Literal

from pydantic import BaseModel

JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


class JobCreate(BaseModel):
    job_id: str
    episode_id: str
    video_uri: str
    stages: list[str] | None = None  # None이면 전체 단계
    options: dict = {}
    callback_url: str | None = None


class JobState(BaseModel):
    job_id: str
    status: JobStatus
    current_stage: str | None = None
    progress: float = 0.0
    error: str | None = None
