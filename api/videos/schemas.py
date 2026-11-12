"""/videos 응답 모양 (api ↔ analyzer 연동 스펙, docs/PLAN.md 6장). api 는 이 모양만 믿는다."""

from pydantic import BaseModel


class VideoInfo(BaseModel):
    name: str                # VIDEOS_ROOT 기준 파일 이름
    size_bytes: int
    duration_sec: float | None
    width: int | None
    height: int | None
    has_audio: bool
    selectable: bool         # 1920x1080 이고 소리가 있으면 true
    reason: str | None = None  # selectable 이 false 인 이유
