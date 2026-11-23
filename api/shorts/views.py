"""/shorts 엔드포인트: 숏폼 렌더 작업 접수·조회.

/jobs(분석)와 달리 다른 작업이 돌고 있어도 거절하지 않고 줄을 세운다. 렌더는 수 초 ~ 수십 초라
기다리게 하는 편이 api 가 다시 보내는 것보다 낫다. 워커 스레드 하나가 순서대로 ffmpeg 를 돌린다.

같은 key 를 다시 보내면
- 같은 내용: 기존 상태를 돌려준다. 실패한 것이면 다시 줄을 세운다 (재시도)
- 다른 내용: 409
- 서버를 다시 띄워 기록이 없어도 결과 파일(shorts/<key>.mp4 + .json)이 있으면 succeeded 로 돌려준다
"""

import json
import logging
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.paths import OUTPUTS, video_path
from api.shorts.schemas import QueryRequest, QueryResult, ShortCreate, ShortState
from pipeline.shorts import bgm
from pipeline.shorts.render import render

log = logging.getLogger(__name__)
router = APIRouter()


@dataclass
class Short:
    spec: ShortCreate
    status: str = "queued"
    attempts: int = 0
    meta: dict | None = None
    error: str | None = None


SHORTS: dict[str, Short] = {}
QUEUE: deque[str] = deque()
_cond = threading.Condition()  # SHORTS · QUEUE 는 이 락 안에서만 읽고 바꾼다
_worker: threading.Thread | None = None


def _out_path(spec: ShortCreate) -> Path:
    return OUTPUTS / Path(spec.video).stem / "shorts" / f"{spec.key}.mp4"


@router.post("/shorts", response_model=ShortState)
def create_short(body: ShortCreate):
    """숏폼 렌더를 줄에 세운다. 404 영상 없음 · 409 같은 key 로 다른 내용 · 400 구간이 시간순이 아님."""
    try:
        video_path(body.video)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    spans = [(s.start_sec, s.end_sec) for s in body.segments]
    if any(e <= s for s, e in spans) or any(b[0] < a[1] for a, b in zip(spans, spans[1:])):
        raise HTTPException(400, "구간은 시간순이고 겹치지 않아야 합니다")
    with _cond:
        short = SHORTS.get(body.key)
        if short is not None:
            if short.spec != body:
                raise HTTPException(409, f"같은 key 로 다른 숏폼이 이미 있습니다: {body.key}")
            if short.status == "failed":
                short.status, short.error = "queued", None
                QUEUE.append(body.key)
                _cond.notify()
            return _state(short)
        done = _from_disk(body)
        if done is not None:
            SHORTS[body.key] = done
            return _state(done)
        SHORTS[body.key] = Short(spec=body)
        QUEUE.append(body.key)
        _ensure_worker()
        _cond.notify()
        return _state(SHORTS[body.key])


@router.post("/shorts/query", response_model=QueryResult)
def query_short(body: QueryRequest):
    """모음형 숏폼 요청 글 → 조건 (LLM, pipeline/shorts/query.py). LLM 을 못 쓰면 error 만 채운다 (200)."""
    from pipeline.shorts import query

    return QueryResult(**query.parse(body.text, body.players))


@router.get("/shorts/{key}", response_model=ShortState)
def get_short(key: str):
    """숏폼 상태. 없는 key 면 404 (서버를 다시 띄웠으면 api 가 같은 내용으로 다시 보내면 된다)."""
    with _cond:
        if key not in SHORTS:
            raise HTTPException(404, "숏폼을 찾을 수 없습니다")
        return _state(SHORTS[key])


def _from_disk(spec: ShortCreate) -> Short | None:
    """결과 파일과 렌더 기록이 이미 있으면 끝난 숏폼으로 본다."""
    out = _out_path(spec)
    meta_path = out.with_suffix(".json")
    if out.exists() and meta_path.exists():
        return Short(spec=spec, status="succeeded", meta=json.loads(meta_path.read_text(encoding="utf-8")))
    return None


def _ensure_worker() -> None:
    global _worker
    if _worker is None or not _worker.is_alive():
        _worker = threading.Thread(target=_work, name="shorts-worker", daemon=True)
        _worker.start()


def _work() -> None:
    """줄에서 하나씩 꺼내 렌더한다. 렌더는 락 밖에서 (그동안에도 접수·조회가 된다)."""
    while True:
        with _cond:
            while not QUEUE:
                _cond.wait()
            short = SHORTS[QUEUE.popleft()]
            short.status, short.attempts = "rendering", short.attempts + 1
            spec = short.spec
        try:
            meta = render(video_path(spec.video), _out_path(spec), [s.model_dump() for s in spec.segments],
                          title_lines=spec.title_lines, captions=[c.model_dump() for c in spec.captions],
                          bgm=bgm.pick(spec.key))
            result = ("succeeded", meta, None)
        except Exception as e:  # ffmpeg 실패, 파일 없음 등. 워커는 멈추지 않는다
            log.exception("숏폼 %s 렌더 실패", spec.key)
            result = ("failed", None, str(e))
        with _cond:
            short.status, short.meta, short.error = result


def _state(short: Short) -> ShortState:
    """숏폼 상태. _cond 안에서 부른다."""
    meta = short.meta or {}
    work_name = Path(short.spec.video).stem
    position = list(QUEUE).index(short.spec.key) if short.status == "queued" and short.spec.key in QUEUE else None
    return ShortState(
        key=short.spec.key, work_name=work_name, status=short.status,
        output=f"{work_name}/shorts/{short.spec.key}.mp4" if short.status == "succeeded" else None,
        duration_sec=meta.get("duration_sec"), render_sec=meta.get("render_sec"), size_bytes=meta.get("size_bytes"),
        attempts=short.attempts, queue_position=position, error=short.error, bgm=meta.get("bgm"))
