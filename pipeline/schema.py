"""파이프라인 데이터 스키마 (초안, schema_version 1).

단위
- ts: 재생용 HLS 조각. 하드컷·전환 경계에서 자르고, 4초보다 길면 균등 분할
- segment: 전환(cut/gradual) 사이의 연속 구간. 일반 ts들의 묶음
- scene: 문맥상 이어지는 segment들의 묶음 (의미 단위, VLM·LLM 분석 단위)

규칙
- 시간은 원본 영상 기준 초 단위 float, 필드명은 *_sec
- id는 회차 안에서 1부터 시작하는 정수
- 파일 경로는 회차 작업 디렉터리(outputs/<영상명>/) 기준 상대 경로

결과 흐름
- main: 회차마다 분석 결과(EpisodeAnalysis)를 저장소(MinIO analysis/{episode}/)에 보관한다.
  인물 정보는 모른다. 대사의 speaker 는 S1, S2 … 라벨뿐이다.
- api: ui 요청을 받을 때 저장된 결과를 읽어 그 자리에서 header(작품·회차·인물·화자 매핑)를 붙인다.
  EpisodeDocument 는 그 결과 모양(회차 하나 = JSON 하나)이며 main 이 만들지 않는다.
- body(scenes, lines) 의 구성은 아직 미정. 아래 Analysis*/Body 는 임시 초안이다.
"""

from datetime import datetime, timezone
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 1

TransitionKind = Literal["cut", "gradual"]  # gradual: 디졸브·와이프·페이드 등 구간이 있는 전환
TsKind = Literal["content", "transition"]
SceneType = Literal["talk", "action", "reaction", "transition", "montage", "mixed"]


class TimeSpan(BaseModel):
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)

    @model_validator(mode="after")
    def _check_order(self):
        if self.end_sec < self.start_sec:
            raise ValueError(f"end_sec({self.end_sec}) < start_sec({self.start_sec})")
        return self


# ---------------------------------------------------------------- 단계 공통 껍데기

R = TypeVar("R", bound=BaseModel)


class StageOutput(BaseModel, Generic[R]):
    """outputs/<영상명>/analysis/<stage>.json"""

    schema_version: int = SCHEMA_VERSION
    stage: str
    episode: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    params: dict = {}  # 이 결과를 만들 때 쓴 설정 (임계값 등)
    result: R


# ---------------------------------------------------------------- preprocess

class VideoInfo(BaseModel):
    codec: str
    width: int
    height: int
    fps: float


class AudioInfo(BaseModel):
    codec: str
    sample_rate: int
    channels: int


class AudioFile(BaseModel):
    path: str  # audio.wav
    sample_rate: int = 16000
    channels: int = 1


class PreprocessResult(BaseModel):
    source_path: str
    duration_sec: float
    video: VideoInfo
    audio: AudioInfo | None  # 오디오 트랙이 없으면 None
    audio_file: AudioFile | None


# ---------------------------------------------------------------- segments (전환 검출)

class Transition(TimeSpan):
    """segment 사이의 전환. cut 은 start_sec == end_sec (한 시점)."""

    id: int
    kind: TransitionKind
    black: bool = False  # 검은 화면을 거침 (암전). 시간 경과·큰 장면 전환 신호


class Segment(TimeSpan):
    id: int
    transition_in: int | None = None   # 앞쪽 전환 id (영상 시작이면 None)
    transition_out: int | None = None  # 뒤쪽 전환 id (영상 끝이면 None)


class SegmentsResult(BaseModel):
    transitions: list[Transition]
    segments: list[Segment]


# ---------------------------------------------------------------- hls

class Ts(TimeSpan):
    id: int
    path: str  # hls/seg_00001.ts
    kind: TsKind
    segment_id: int | None = None   # content면 소속 segment
    transition_id: int | None = None  # transition이면 해당 전환

    @model_validator(mode="after")
    def _check_owner(self):
        if self.kind == "content" and (self.segment_id is None or self.transition_id is not None):
            raise ValueError("content ts는 segment_id만 가진다")
        if self.kind == "transition" and (self.transition_id is None or self.segment_id is not None):
            raise ValueError("transition ts는 transition_id만 가진다")
        return self


class HlsResult(BaseModel):
    playlist_path: str  # hls/index.m3u8
    max_ts_sec: float = 4.0
    ts: list[Ts]


# ---------------------------------------------------------------- keyframes

class Keyframe(BaseModel):
    segment_id: int
    time_sec: float
    path: str  # frames/seg_0001.jpg


class KeyframesResult(BaseModel):
    keyframes: list[Keyframe]


# ---------------------------------------------------------------- asr

class Word(TimeSpan):
    text: str
    prob: float | None = None


class Line(TimeSpan):
    """한 사람이 한 번 말한 대사. segment 경계를 넘어도 자르지 않는다."""

    id: int
    speaker: str | None = None  # S1, S2 … (화자 분리 전이면 None)
    language: str | None = None  # 대사별 감지 언어 (ko, ja, en …)
    text: str
    words: list[Word] = []


class AsrResult(BaseModel):
    lines: list[Line]


# ---------------------------------------------------------------- similarity

class SegmentPair(BaseModel):
    """인접 segment 사이의 연속성 점수 (scene 병합 입력)."""

    prev_id: int
    next_id: int
    visual: float  # MobileCLIP 화면 유사도
    dialogue: float | None = None  # 대사가 경계를 넘는지 등
    audio: float | None = None


class SimilarityResult(BaseModel):
    pairs: list[SegmentPair]


# ---------------------------------------------------------------- scenes (병합)

class Scene(TimeSpan):
    id: int
    segment_ids: list[int]


class ScenesResult(BaseModel):
    scenes: list[Scene]


# ---------------------------------------------------------------- vlm

class SceneAnalysis(BaseModel):
    scene_id: int
    type: SceneType
    summary: str
    keywords: list[str] = []
    characters: list[int] = []  # header.characters 의 id
    keyframe_paths: list[str] = []  # VLM 에 넣은 대표 프레임 (3~5장)


class VlmResult(BaseModel):
    backend: str  # mock / mlx / transformers / vllm
    model: str | None = None
    scenes: list[SceneAnalysis]


# ---------------------------------------------------------------- 분석 결과 (main → api)

class AnalysisScene(TimeSpan):
    id: int
    type: SceneType
    summary: str
    keywords: list[str] = []
    segment_ids: list[int] = []
    speakers: list[str] = []  # 이 scene 에서 말한 화자 라벨
    black_before: bool = False  # 암전 뒤에 시작하는 scene


class AnalysisLine(TimeSpan):
    id: int
    scene_id: int | None  # 시작 시각이 속한 scene
    speaker: str | None  # S1, S2 … (화자 분리 전이면 None)
    language: str | None = None  # ko, ja, en … (다국어 대사)
    text: str


class SpeakerSummary(BaseModel):
    """api 의 화자 매핑 화면용."""

    speaker: str
    line_count: int
    total_sec: float
    samples: list[str] = []  # 대표 대사 몇 개


class EpisodeAnalysis(BaseModel):
    """outputs/<영상명>/analysis/episode.json. 회차 하나의 최종 분석 결과."""

    schema_version: int = SCHEMA_VERSION
    episode: str
    duration_sec: float
    scenes: list[AnalysisScene]
    lines: list[AnalysisLine]
    speakers: list[SpeakerSummary]


# ---------------------------------------------------------------- LLM 입력 문서 (회차별 header + body)
# 인물·제작진 정보는 api 가 관리하므로 main 은 id, name 만 요구하고 나머지 필드는 그대로 통과시킨다.

class Loose(BaseModel):
    model_config = ConfigDict(extra="allow")


class Character(Loose):
    id: int
    name: str  # 예: name_en, actor, aliases, episode_count 등은 api 가 자유롭게 추가


class Crew(Loose):
    name: str
    job: str  # Director, Writer …


class SeriesInfo(Loose):
    title: str
    synopsis: str | None = None
    crew: list[Crew] = []


class EpisodeInfo(Loose):
    number: int
    title: str | None = None
    synopsis: str | None = None
    air_date: str | None = None


class Header(BaseModel):
    """LLM 에 함께 전달하는 정보. fields 는 body 각 필드의 설명."""

    series: SeriesInfo
    episode: EpisodeInfo
    duration_sec: float
    characters: list[Character] = []
    speakers: dict[str, int | None] = {}  # 이 회차의 화자 라벨 → characters.id
    fields: dict[str, str] = {}


class Body(BaseModel):
    scenes: list[AnalysisScene]
    lines: list[AnalysisLine]


class EpisodeDocument(BaseModel):
    """api 가 요청 시점에 만드는 회차 문서 (header + body). 회차 하나 = JSON 하나."""

    schema_version: int = SCHEMA_VERSION
    header: Header
    body: Body


FIELD_DESCRIPTIONS = {
    "start_sec/end_sec": "원본 영상 기준 시작·끝 시각(초)",
    "header.speakers": "화자 라벨(S1 …) → header.characters 의 id. 회차마다 라벨이 다르다",
    "scenes": "문맥상 이어지는 장면 단위",
    "scenes[].type": "talk=대화 중심, action=행동·사건, reaction=표정·반응, "
    "transition=시간 변화·회상 진입, montage=짧은 장면 연결, mixed=혼합",
    "scenes[].black_before": "암전 뒤에 시작하는 장면 (시간 경과·큰 전환일 가능성)",
    "lines": "대사. speaker 는 화자 라벨, language 는 대사 언어",
}
