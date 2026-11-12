"""/jobs 엔드포인트: 분석 작업 시작·조회·취소.

작업은 이 컨테이너 안에서 `python -m pipeline` 프로세스로 돌린다 (한 번에 하나).
큐(Redis 등)는 아직 없어서 서버를 다시 띄우면 진행 중이던 작업 기록은 사라진다 (결과 파일은 남는다).
"""

import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.jobs.schemas import JobCreate, JobState
from api.paths import OUTPUTS, video_path
from pipeline.stages import IMPLEMENTED

log = logging.getLogger(__name__)
router = APIRouter()

_STAGE_LINE = re.compile(r"^\[(\w+)\] 시작")
MAX_FINISHED = 100  # 끝난 작업 기록은 이만큼만 남긴다 (오래된 것부터 지움)
_ERROR_TAIL_BYTES = 4096  # 실패하면 로그 끝부분에서 에러를 읽는다


@dataclass
class Job:
    spec: JobCreate
    proc: subprocess.Popen
    log_path: Path
    started: float = field(default_factory=time.monotonic)
    ended: float | None = None
    cancelled: bool = False
    log_pos: int = 0                  # 로그에서 여기까지 읽었다
    current_stage: str | None = None  # 로그의 마지막 "[단계] 시작"
    error: str | None = None


JOBS: dict[str, Job] = {}
_lock = threading.Lock()  # JOBS 와 Job 상태는 이 락 안에서만 읽고 바꾼다


@router.post("/jobs", response_model=JobState)
def create_job(body: JobCreate):
    """분석 작업을 시작한다. 같은 요청(job_id·내용 동일)을 다시 보내면 기존 상태를 돌려준다.

    400 실행할 수 없는 단계 · 404 영상 없음 · 409 같은 job_id 로 다른 작업 / 다른 작업 진행 중 · 500 프로세스 시작 실패
    """
    with _lock:
        if body.job_id in JOBS:
            job = JOBS[body.job_id]
            if job.spec != body:
                raise HTTPException(409, f"같은 job_id 로 다른 작업이 이미 있습니다: {body.job_id}")
            return _state(job)
        unknown = [s for s in body.stages if s not in IMPLEMENTED]
        if not body.stages or unknown:
            raise HTTPException(400, f"실행할 수 없는 단계: {unknown or '없음'}")
        try:
            video = video_path(body.video)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        if any(j.proc.poll() is None for j in JOBS.values()):
            raise HTTPException(409, "다른 분석이 진행 중입니다")

        log_path = OUTPUTS / video.stem / "work" / "job.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "pipeline", str(video), "--work-dir", str(OUTPUTS), "--only", ",".join(body.stages)]
        for stage, values in body.params.items():
            for key, value in values.items():
                cmd += ["--set", f"{stage}.{key}={json.dumps(value, ensure_ascii=False)}"]
        with log_path.open("w", encoding="utf-8") as logf:
            try:
                # 새 프로세스 그룹: 취소할 때 pipeline 이 띄운 ffmpeg 등까지 한 번에 멈춘다
                proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True,
                                        env={**os.environ, "PYTHONUNBUFFERED": "1"})
            except OSError as e:
                raise HTTPException(500, f"분석 프로세스를 시작하지 못했습니다: {e}") from e
        JOBS[body.job_id] = Job(spec=body, proc=proc, log_path=log_path)
        _prune()
        log.info("job %s 시작: %s", body.job_id, " ".join(cmd))
        return _state(JOBS[body.job_id])


@router.get("/jobs/{job_id}", response_model=JobState)
def get_job(job_id: str):
    """작업 상태. 없는 job_id 면 404 (서버를 다시 띄웠거나 오래된 기록이 지워짐)."""
    with _lock:
        return _state(_get(job_id))


@router.post("/jobs/{job_id}/cancel", response_model=JobState)
def cancel_job(job_id: str):
    """작업을 멈춘다 (pipeline 과 그 하위 프로세스 모두)."""
    with _lock:
        job = _get(job_id)
        if job.proc.poll() is None:
            job.cancelled = True
            try:
                os.killpg(job.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass  # 그 사이에 끝남
        return _state(job)


def _get(job_id: str) -> Job:
    if job_id not in JOBS:
        raise HTTPException(404, "작업을 찾을 수 없습니다")
    return JOBS[job_id]


def _prune() -> None:
    """끝난 작업이 MAX_FINISHED 개를 넘으면 오래된 것부터 지운다."""
    finished = [k for k, j in JOBS.items() if j.proc.poll() is not None]
    for k in finished[:max(0, len(finished) - MAX_FINISHED)]:
        del JOBS[k]


def _read_new_lines(job: Job) -> None:
    """로그에서 지난번 이후 새로 붙은 줄만 읽어 현재 단계를 갱신한다."""
    if not job.log_path.exists():
        return
    with job.log_path.open("rb") as f:
        f.seek(job.log_pos)
        chunk = f.read()
    end = chunk.rfind(b"\n") + 1  # 아직 덜 쓴 마지막 줄은 다음에 읽는다
    job.log_pos += end
    for line in chunk[:end].decode("utf-8", errors="replace").splitlines():
        if m := _STAGE_LINE.match(line):
            job.current_stage = m.group(1)


def _log_tail(job: Job, lines: int = 5) -> str:
    if not job.log_path.exists():
        return ""
    with job.log_path.open("rb") as f:
        f.seek(max(0, job.log_path.stat().st_size - _ERROR_TAIL_BYTES))
        text = f.read().decode("utf-8", errors="replace")
    return "\n".join(text.strip().splitlines()[-lines:])


def _state(job: Job) -> JobState:
    """작업 상태. _lock 안에서 부른다."""
    code = job.proc.poll()
    if job.ended is None:
        _read_new_lines(job)
        if code is not None:
            job.ended = time.monotonic()
            if code != 0 and not job.cancelled:
                job.error = _log_tail(job) or f"종료 코드 {code}"
    if code is None:
        status = "running"
    elif job.cancelled:
        status = "cancelled"
    elif code == 0:
        status = "succeeded"
    else:
        status = "failed"
    return JobState(job_id=job.spec.job_id, video=job.spec.video, work_name=Path(job.spec.video).stem,
                    stages=job.spec.stages, status=status, current_stage=job.current_stage,
                    elapsed_sec=round((job.ended or time.monotonic()) - job.started, 1), error=job.error)
