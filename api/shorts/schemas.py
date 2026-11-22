"""/shorts 요청·응답 모양 (api ↔ analyzer 연동 스펙, docs/PLAN.md 6장). api 는 이 모양만 믿는다."""

from typing import Literal

from pydantic import BaseModel, Field

ShortStatus = Literal["queued", "rendering", "succeeded", "failed"]


class Span(BaseModel):
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)


class Subtitle(Span):
    text: str


class ShortCreate(BaseModel):
    key: str = Field(pattern=r"^[a-z0-9-]{8,64}$")  # 결과 파일 이름. api 가 입력 해시로 정한다 (같은 입력 = 같은 key)
    video: str                                     # VIDEOS_ROOT 기준 파일 이름
    segments: list[Span] = Field(min_length=1, max_length=5)  # 원본 기준, 시간순·겹치지 않게
    subtitles: list[Subtitle] = []                 # 원본 기준 대사 (api 가 검수로 고친 문장을 넣어 보냄)
    title: str | None = None                       # 숏폼 이름 (렌더 기록·목록용, 영상에는 그리지 않음)


class ShortState(BaseModel):
    key: str
    work_name: str                # outputs/<work_name>/shorts/<key>.mp4
    status: ShortStatus
    output: str | None = None     # 결과 mp4 (OUTPUTS 기준 상대 경로). succeeded 일 때
    duration_sec: float | None = None
    render_sec: float | None = None  # ffmpeg 렌더에 걸린 시간
    size_bytes: int | None = None
    attempts: int = 0             # 이 서버에서 렌더를 시도한 횟수
    queue_position: int | None = None  # queued 일 때 앞에 남은 작업 수
    error: str | None = None
