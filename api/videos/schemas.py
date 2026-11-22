"""/videos 응답 모양 (api ↔ analyzer 연동 스펙, docs/PLAN.md 6장). api 는 이 모양만 믿는다.

GET /videos/{name}/file 은 JSON 이 아니라 원본 영상 파일 그대로다 (Range 요청 가능, 없으면 404).
api 는 이 주소를 GET /games/{id} 의 video_url 로 ui 에 넘긴다.
"""

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
