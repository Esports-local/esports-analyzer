"""/shorts 요청·응답 모양 (api ↔ analyzer 연동 스펙, docs/PLAN.md 6장). api 는 이 모양만 믿는다."""

from typing import Literal

from pydantic import BaseModel, Field

ShortStatus = Literal["queued", "rendering", "succeeded", "failed"]


class Span(BaseModel):
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)


class Caption(BaseModel):
    start_sec: float = Field(ge=0)   # 원본 영상 기준
    end_sec: float = Field(ge=0)
    lines: list[str] = Field(min_length=1, max_length=2)


class ShortCreate(BaseModel):
    key: str = Field(pattern=r"^[a-z0-9-]{8,64}$")  # 결과 파일 이름. api 가 입력 해시로 정한다 (같은 입력 = 같은 key)
    video: str                                     # VIDEOS_ROOT 기준 파일 이름
    segments: list[Span] = Field(min_length=1)  # 원본 기준, 시간순·겹치지 않게 (개수·길이 제한 없음)
    title: str | None = None                       # 숏폼 이름 (렌더 기록·목록용)
    title_lines: list[str] = Field(default=[], max_length=2)  # 영상 위 제목 (첫 줄 흰색, 둘째 줄 노란색)
    captions: list[Caption] = []                   # 게임 화면 아래쪽 설명 자막 (원본 기준 시각)


class QueryRequest(BaseModel):
    """모음형 숏폼 요청 글 → 조건 (POST /shorts/query)."""
    text: str = Field(min_length=1, max_length=200)
    players: dict[str, str]                        # {슬롯: "팀 이름 (챔피언)"} 이 경기 선수


class QueryResult(BaseModel):
    filter: dict[str, str] = {}   # {killer: 슬롯, type: kill|objective|tower, team: blue|red, tag: first_blood|multikill|teamfight}
    title: str | None = None      # 10자 이내 제목 제안
    note: str | None = None       # 조건으로 못 나타낸 부분
    error: str | None = None      # LLM 을 못 썼으면 이유 (filter 는 비어 있음)


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
    bgm: dict | None = None       # 깐 BGM {title, source, license, attribution} (succeeded, 곡이 있었을 때)
