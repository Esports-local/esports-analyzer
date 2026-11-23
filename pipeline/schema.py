"""파이프라인 데이터 스키마 (schema_version 1).

단위
- segment: 장면 전환(cut · dissolve · wipe) 사이의 연속 구간. 화면 종류(live · replay · postgame · break)가 붙는다
- line: 해설 대사 한 줄 (asr)
- OcrRun: 중계 화면 고정 영역(tag + slot)에서 같은 값이 이어진 구간 (roi_ocr)

규칙
- 시간은 원본 영상 기준 초 단위 float, 필드명은 *_sec
- id는 영상 안에서 1부터 시작하는 정수
- 파일 경로는 작업 디렉터리(outputs/<영상명>/) 기준 상대 경로

파일 구성 (outputs/<영상명>/)
- work/<stage>.json   단계 사이에 넘기는 중간 결과. 재실행·디버그용
- shorts/<key>.mp4    숏폼 렌더 결과 (옆에 렌더 기록 .json)
"""

from datetime import datetime, timezone
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 1

ScreenKind = Literal["live", "replay", "postgame", "countdown", "draft", "break"]
TransitionKind = Literal["cut", "dissolve", "wipe"]  # dissolve: 서서히 겹침·페이드, wipe: 한쪽부터 쓸고 지나감


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
    range_start_sec: float = 0.0  # 일부 구간만 분석했으면 그 시작 (audio.wav 0초 = 원본 이 초)


# ---------------------------------------------------------------- segments (장면 전환 + 화면 종류)

class Transition(TimeSpan):
    """segment 사이의 전환. cut 은 start_sec == end_sec (한 시점)."""

    id: int
    kind: TransitionKind
    black: bool = False  # 검은 화면을 거침 (암전)


class Segment(TimeSpan):
    id: int
    transition_in: int | None = None   # 앞쪽 전환 id (영상 시작이면 None)
    transition_out: int | None = None  # 뒤쪽 전환 id (영상 끝이면 None)
    kind: ScreenKind | None = None   # live 경기 · replay 리플레이·하이라이트 · postgame 경기 뒤 통계 · countdown 세트 사이 · draft 밴픽 · break 그 밖 (screens 규칙 있을 때)
    game_start: str | None = None    # live 면 안에서 처음 읽은 경기 시간 "12:34"
    game_end: str | None = None      # live 면 안에서 마지막에 읽은 경기 시간


class SegmentsResult(BaseModel):
    layout: str | None = None        # 쓴 레이아웃 파일
    sample_sec: float | None = None  # 판별 간격 (구간 경계 오차는 이것의 절반)
    samples: dict[str, int] = {}     # 종류별 판별 화면 수
    transitions: list[Transition] = []
    segments: list[Segment] = []
    game: dict | None = None         # 경기 구간 {start_sec, end_sec, last_clock, header_length_sec, postgame?, checks?}


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
    text: str                    # 고유명사 교정을 거친 문장 (pipeline/audio/terms.py)
    raw: str | None = None       # whisper 원문 (교정으로 바뀐 게 있을 때만)
    fixes: list[dict] = []       # [{from, to, kind, slot, score}] 바꾼 내역


class AsrResult(BaseModel):
    lines: list[Line]


# ---------------------------------------------------------------- timeline (화면 숫자 변화 → 경기 이벤트)

EventType = Literal["kill", "tower", "objective"]


class TimelineEvent(TimeSpan):
    """경기 이벤트. 숏폼 구간의 기준점.

    source=opgg: op.gg 기록(경기 시각)을 화면 경기 시계로 영상 초로 옮긴 것. start_sec = end_sec = 그 시각.
    source=ocr: 화면 숫자(팀 킬·타워·오브젝트 수)가 오른 시점 (op.gg 가 없을 때, 타워는 항상).
      start_sec: 이벤트가 일어났을 수 있는 가장 이른 시각 (앞 값이 마지막으로 읽힌 때).
      end_sec: 새 값이 처음 읽힌 시각. 화면이 계속 보였으면 둘이 거의 같다.
    """

    id: int
    type: EventType
    subtype: str | None = None    # objective: grubs · dragon:<종류> · elder · baron · herald · atakhan (화면에서 알 수 있을 때)
    team: str | None              # 진영 슬롯 (blue / red), 모르면 None
    team_name: str | None = None  # 화면에서 읽은 팀 이름
    value_before: int
    value_after: int
    game_time: str | None = None  # 새 값이 읽힌 시점의 경기 시계 (mm:ss)
    players: dict[str, list[str]] = {}  # {"killer": [...], "victims": [...], "assists": [...]} KDA 변화로 찾은 것만
    line_ids: list[int] = []      # 앞뒤 대사 (asr)
    evidence: list[str] = []      # 근거 ("team_kills/red 2→3 @386.0", "opgg kill 15:10")
    source: Literal["opgg", "ocr"] = "ocr"
    position: dict | None = None  # op.gg 맵 좌표 {x, y}
    ocr_lag_sec: float | None = None  # op.gg 킬: 화면 팀 킬 수가 오른 시각 - 이 시각 (동기화 검증, 정상 1~2초)


class TimelineResult(BaseModel):
    events: list[TimelineEvent]
    teams: dict[str, str] = {}    # {진영: 팀 이름}
    players: dict[str, str] = {}  # {진영_포지션: 선수 이름}
    source: Literal["opgg", "ocr"] = "ocr"  # 킬·오브젝트를 어디서 만들었나 (타워는 항상 ocr)
    unmapped: list[str] = []      # 영상 초로 옮기지 못한 op.gg 기록 (분석 범위 밖 제외) "kill 15:10"
    sync: dict = {}               # 시계 동기화 검증 {checked, matched, lag_median_sec, lag_min_sec, lag_max_sec, outliers}


# ---------------------------------------------------------------- header

class HeaderResult(BaseModel):
    """세트 영상의 경기 기본 정보 (pipeline/header/). op.gg·Data Dragon 만으로 만든다. 받지 못하면 error 만 채운다."""

    match: dict = {}
    teams: dict = {}
    players: list[dict] = []
    bans: dict = {}
    result: dict = {}
    series: dict = {}
    source: dict = {}
    error: str | None = None


# ---------------------------------------------------------------- opgg_timeline (header 단계가 같이 만듦, 정답지)

class OpggTimelineResult(BaseModel):
    """op.gg 이 세트 기록 (pipeline/header/opgg.py parse_timeline). 시각은 경기 시각(초), 좌표는 맵 좌표.

    바디(context)는 쓰지 않고, 바디를 검증하는 정답지로만 쓴다. 받지 못하면 error 만 채운다.
    """

    opgg_game_id: int | None = None
    detailed: bool = False        # op.gg 상세 기록 여부 (false 면 아래 목록이 비어 있을 수 있음)
    length_sec: float | None = None
    objectives: list[dict] = []   # [{t, game_time, type, subtype, side, killer, position}]
    kills: list[dict] = []        # [{t, game_time, side, killer, position}]
    positions: dict = {}          # {슬롯: [[t, x, y], ...]} 1분마다
    gold: dict = {}               # {진영: [[t, 골드], ...]} 1분마다
    totals: dict = {}             # {진영: {towers, inhibitors, dragons, elders, barons, heralds, atakhans}}
    error: str | None = None


# ---------------------------------------------------------------- context (바디, pipeline/game/context.py)

class ContextResult(BaseModel):
    """영상 기준 경기 데이터 (바디). 사실만 담고, 하이라이트 판단은 숏폼 쪽에서 한다."""

    game: dict = {}             # {start_sec, end_sec, length_sec, winner, checks}
    players: dict = {}          # {슬롯: {team, name, champion, champion_en}}
    clock: list[dict] = []      # [{start_sec, end_sec, game_start_sec}] 경기 시간이 흐르는 구간
    segments: list[dict] = []   # [{id, start_sec, end_sec, kind, transition_in, game_time}]
    events: list[dict] = []     # [{id, t, start_sec, end_sec, game_time, type, team, killer, victims, assists, tags, segment, line_ids, source}]
    signals: dict = {}          # {step_sec, start_sec, loudness, gold_diff, cut_rate, kind}


# ---------------------------------------------------------------- shorts (숏폼 후보, pipeline/shorts/scenes.py)

class ShortCandidate(BaseModel):
    """대사로 이은 컷 묶음(장면) 중 숏폼 후보 하나. 구간은 장면 처음~끝 그대로."""

    id: str                     # c1, c2 … (시간순)
    cuts: list[str]             # analyzer.json 컷 id
    kind: str | None = None
    start_sec: float
    end_sec: float
    duration_sec: float
    events: list[dict] = []     # [{id, t, type, subtype, team, tags}]
    lines: list[str] = []       # 장면 안 해설 문장 (시간순)


class ShortsResult(BaseModel):
    scenes: int = 0             # 묶은 장면 수 (후보가 아닌 것 포함)
    candidates: list[ShortCandidate] = []
