"""파이프라인 데이터 스키마 (schema_version 1).

단위
- ts: 재생용 HLS 조각. 하드컷·전환 경계에서 자르고, 4초보다 길면 균등 분할
- segment: 전환(cut/gradual) 사이의 연속 구간 (중계 화면 전환: 경기 화면 ↔ 캠·리플레이 등)
- line: 해설 대사 한 줄 (asr)
- OcrRun: 중계 화면 고정 영역(tag + slot)에서 같은 값이 이어진 구간 (roi_ocr)

규칙
- 시간은 원본 영상 기준 초 단위 float, 필드명은 *_sec
- id는 영상 안에서 1부터 시작하는 정수
- 파일 경로는 작업 디렉터리(outputs/<영상명>/) 기준 상대 경로

파일 구성 (outputs/<영상명>/)
- work/<stage>.json   단계 사이에 넘기는 중간 결과. 재실행·디버그용
- hls/index.m3u8      재생용. ts 목록·길이, segment·전환 표시
                        #EXTINF:<길이>,seg=<segment id>
                        #EXTINF:<길이>,transition=gradual[ black]   (cut 은 한 시점이라 ts 없음)
                        master.m3u8: RESOLUTION, FRAME-RATE, CODECS, BANDWIDTH
"""

from datetime import datetime, timezone
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 1

TransitionKind = Literal["cut", "gradual"]  # gradual: 디졸브·와이프·페이드 등 구간이 있는 전환


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


# ---------------------------------------------------------------- roi_ocr (중계 화면 고정 영역 OCR)

class OcrRun(TimeSpan):
    """한 영역(tag + slot)에서 같은 값이 이어진 구간. 읽지 못한(검증 실패) 구간은 기록하지 않는다."""

    tag: str               # 설정 파일의 tag (player_name, kda, team …)
    slot: str              # 같은 tag 안의 위치 (blue_top, red …). 없으면 ""
    kind: str | None = None  # team / player / score / time …
    text: str              # 최종 값 (사전 대조로 고친 정식 이름이면 그 이름)
    raw: str               # OCR 이 읽은 그대로 (허용 문자 정리 전)
    conf: float            # 구간 안 OCR 신뢰도 평균
    samples: int           # 구간에 들어간 샘플 프레임 수


class RoiOcrResult(BaseModel):
    layout: str            # 설정 파일 경로
    sample_fps: float
    base_resolution: tuple[int, int]
    rois: int              # 설정의 영역 수
    frames: int            # 읽은 샘플 프레임 수
    ocr_calls: int         # 실제로 OCR 을 돌린 횟수 (화면이 그대로면 앞 값을 다시 씀)
    runs: list[OcrRun]


# ---------------------------------------------------------------- asr

class Line(TimeSpan):
    """한 사람이 한 번 말한 대사. segment 경계를 넘어도 자르지 않는다."""

    id: int
    speaker: str | None = None  # S1, S2 … (화자 분리 전이면 None)
    language: str | None = None  # 대사별 감지 언어 (ko, ja, en …)
    text: str


class AsrResult(BaseModel):
    lines: list[Line]
