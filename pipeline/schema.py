"""파이프라인 데이터 스키마 (초안, schema_version 1).

단위
- ts: 재생용 HLS 조각. 하드컷·전환 경계에서 자르고, 4초보다 길면 균등 분할
- segment: 전환(cut/gradual) 사이의 연속 구간. 일반 ts들의 묶음
- scene: 문맥상 이어지는 segment들의 묶음 (의미 단위, VLM·LLM 분석 단위)

규칙
- 시간은 원본 영상 기준 초 단위 float, 필드명은 *_sec
- id는 회차 안에서 1부터 시작하는 정수
- 파일 경로는 회차 작업 디렉터리(outputs/<영상명>/) 기준 상대 경로

파일 구성 (outputs/<영상명>/)
- work/<stage>.json   단계 사이에 넘기는 중간 결과. 재실행·디버그용, 외부에 공개하지 않음
- hls/index.m3u8      ▶ 결과물: 영상 구조. ts 목록·길이, segment·전환 표시
                        #EXTINF:<길이>,seg=<segment id>
                        #EXTINF:<길이>,transition=gradual[ black]   (cut 은 한 시점이라 ts 없음)
                        master.m3u8: RESOLUTION, FRAME-RATE, CODECS, BANDWIDTH
- analysis/episode.json ▶ 결과물: 영상 내용 (EpisodeAnalysis). 씬·지문·대사·얼굴
  MinIO 에는 hls/ 와 analysis/ 만 올린다.

결과 흐름
- main: 회차마다 hls/ 와 analysis/ 를 저장소(MinIO)에 보관한다.
  인물 정보는 모른다. 대사의 speaker 는 S1 …, 화면의 얼굴은 F1 … 라벨뿐이다.
  대사는 들린 그대로의 원문. 번역·해석은 api 에서 한다.
- api: ui 요청을 받을 때 저장된 결과를 읽어 그 자리에서 header(작품·회차·인물·화자/얼굴 매핑)를 붙인다.
  EpisodeDocument 는 그 결과 모양(회차 하나 = JSON 하나)이며 main 이 만들지 않는다.
- body 는 대본처럼 scene 단위, scene 안에 대사(lines). 세부 필드는 실제 영상으로 결과를 보며 조정한다.
"""

from datetime import datetime, timezone
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 1

TransitionKind = Literal["cut", "gradual"]  # gradual: 디졸브·와이프·페이드 등 구간이 있는 전환
SceneType = Literal["talk", "action", "reaction", "transition", "montage", "mixed"]


class TimeSpan(BaseModel):
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)

    @model_validator(mode="after")
    def _check_order(self):
        """end_sec 가 start_sec 보다 앞이면 거부한다."""
        if self.end_sec < self.start_sec:
            raise ValueError(f"end_sec({self.end_sec}) < start_sec({self.start_sec})")
        return self


# ---------------------------------------------------------------- 단계 공통 껍데기

R = TypeVar("R", bound=BaseModel)


class StageOutput(BaseModel, Generic[R]):
    """outputs/<영상명>/work/<stage>.json (중간 결과)"""

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
    fps: float  # 평균 프레임레이트. 분석 단계는 이 값으로 CFR 에 맞춰 읽는다
    vfr: bool  # 가변 프레임레이트 여부 (화면 녹화본에서 흔함)


class AudioInfo(BaseModel):
    codec: str
    sample_rate: int
    channels: int


class AudioFile(BaseModel):
    path: str  # work/audio.wav
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
# JSON 대신 hls/index.m3u8 이 결과다 (모듈 docstring 의 #EXTINF 표시 참고).


# ---------------------------------------------------------------- keyframes

class Keyframe(BaseModel):
    segment_id: int
    time_sec: float
    path: str  # frames/seg_0001.jpg


class KeyframesResult(BaseModel):
    keyframes: list[Keyframe]


# ---------------------------------------------------------------- faces (InsightFace)

class FaceDetection(BaseModel):
    id: int
    segment_id: int
    time_sec: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (프레임 크기 기준 0~1)
    det_score: float
    face_id: str | None  # 군집 라벨 F1, F2 … (작은 군집이면 None)


class FaceCluster(BaseModel):
    """회차 안에서 같은 사람으로 묶인 얼굴. 라벨은 회차마다 새로 붙는다."""

    face_id: str
    count: int
    segment_ids: list[int]
    best_detection_id: int  # 대표 얼굴 (썸네일, 배우 사진 비교용)


class FacesResult(BaseModel):
    embeddings_path: str  # work/faces_embeddings.npy (detections 와 같은 순서의 행, 512차원 정규화)
    clusters: list[FaceCluster]
    detections: list[FaceDetection]


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

class AnalysisLine(TimeSpan):
    """대본의 대사 한 줄 = 한 화자가 이어서 말한 덩어리 전체 (컷·ts 경계를 넘어도 자르지 않음).

    start_sec/end_sec 는 대사 구간 (근거·대사 클릭 시 이동). 결과를 보고 뺄지 정한다.
    """

    speaker: str | None  # S1, S2 … (화자 분리 전이면 None)
    text: str  # 들린 그대로의 원문


class AnalysisScene(TimeSpan):
    """대본의 씬 하나. 대사는 시작 시각이 속한 씬에 시간순으로 들어간다."""

    id: int
    type: SceneType
    summary: str  # 지문 (VLM)
    keywords: list[str] = []
    black_before: bool = False  # 암전 뒤에 시작하는 scene
    faces: list[str] = []  # 화면에 나온 얼굴 라벨 F1, F2 …
    lines: list[AnalysisLine] = []


class SpeakerSummary(BaseModel):
    """api 의 화자 매핑 화면용."""

    speaker: str
    line_count: int
    total_sec: float
    samples: list[str] = []  # 대표 대사 몇 개


class EpisodeAnalysis(BaseModel):
    """outputs/<영상명>/analysis/episode.json. 회차 하나의 최종 분석 결과 (영상 내용만)."""

    schema_version: int = SCHEMA_VERSION
    episode: str
    duration_sec: float
    scenes: list[AnalysisScene]
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
    faces: dict[str, int | None] = {}  # 이 회차의 얼굴 라벨 → characters.id
    fields: dict[str, str] = {}


class Body(BaseModel):
    scenes: list[AnalysisScene]


class EpisodeDocument(BaseModel):
    """api 가 요청 시점에 만드는 회차 문서 (header + body). 회차 하나 = JSON 하나."""

    schema_version: int = SCHEMA_VERSION
    header: Header
    body: Body


FIELD_DESCRIPTIONS = {
    "start_sec/end_sec": "원본 영상 기준 시작·끝 시각(초)",
    "header.speakers": "화자 라벨(S1 …) → header.characters 의 id. 회차마다 라벨이 다르다",
    "header.faces": "얼굴 라벨(F1 …) → header.characters 의 id. 회차마다 라벨이 다르다",
    "scenes": "대본의 씬. 문맥상 이어지는 장면 단위이며 대사(lines)를 시간순으로 포함",
    "scenes[].summary": "지문. 이 씬에서 화면에 보이는 상황",
    "scenes[].faces": "화면에 나온 얼굴 라벨(F1 …). header.faces 로 인물 확인",
    "scenes[].type": "talk=대화 중심, action=행동·사건, reaction=표정·반응, "
    "transition=시간 변화·회상 진입, montage=짧은 장면 연결, mixed=혼합",
    "scenes[].black_before": "암전 뒤에 시작하는 장면 (시간 경과·큰 전환일 가능성)",
    "scenes[].lines": "대사(원문). speaker 는 화자 라벨(header.speakers 로 인물 확인), start_sec/end_sec 는 대사 구간",
}
