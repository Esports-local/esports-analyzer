"""analyzer HTTP 서버. api 가 영상 목록을 보고, 분석 작업을 맡기고, 상태를 묻는다.

작업은 이 컨테이너 안에서 `python -m pipeline` 프로세스로 돌린다 (한 번에 하나).
큐(Redis 등)는 아직 없어서 서버를 다시 띄우면 진행 중이던 작업 기록은 사라진다 (결과 파일은 남는다).
"""

import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException

from app.schemas import EstimateRequest, EstimateResult, JobCreate, JobState, VideoInfo
from pipeline.stages import IMPLEMENTED, STAGES

log = logging.getLogger(__name__)
app = FastAPI(title="esports-analyzer")

VIDEOS_ROOT = Path(os.environ.get("VIDEOS_ROOT", "/data"))
OUTPUTS = Path(os.environ.get("OUTPUTS_ROOT", "outputs"))
VIDEO_EXTS = {".mp4", ".mov", ".mkv"}
_STAGE_LINE = re.compile(r"^\[(\w+)\] 시작")


@dataclass
class Job:
    spec: JobCreate
    proc: subprocess.Popen
    log_path: Path
    started: float = field(default_factory=time.monotonic)
    ended: float | None = None
    cancelled: bool = False


JOBS: dict[str, Job] = {}
_lock = threading.Lock()
_probe_cache: dict[tuple, dict] = {}


@app.get("/health")
def health():
    """서버 상태와 단계 목록."""
    return {"status": "ok", "gpu": False, "stages": STAGES, "implemented": list(IMPLEMENTED)}


@app.get("/videos", response_model=list[VideoInfo])
def list_videos():
    """VIDEOS_ROOT 의 영상 목록. 1920x1080 이고 소리가 있는 영상만 selectable."""
    if not VIDEOS_ROOT.is_dir():
        raise HTTPException(500, f"영상 폴더가 없습니다: {VIDEOS_ROOT}")
    out = []
    for path in sorted(VIDEOS_ROOT.iterdir()):
        if path.suffix.lower() not in VIDEO_EXTS or path.name.startswith("."):
            continue
        info = probe(path)
        reason = None
        if (info["width"], info["height"]) != (1920, 1080):
            reason = f"{info['width']}x{info['height']} (1920x1080 만 가능)"
        elif not info["has_audio"]:
            reason = "소리 트랙 없음"
        out.append(VideoInfo(name=path.name, size_bytes=path.stat().st_size, selectable=reason is None,
                             reason=reason, **info))
    return out


def probe(path: Path) -> dict:
    """ffprobe 로 길이·해상도·소리 유무. (경로, 크기, 수정 시각)으로 캐시."""
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime)
    if key not in _probe_cache:
        res = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height",
                              "-of", "json", str(path)], capture_output=True, text=True)
        data = json.loads(res.stdout or "{}")
        video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
        dur = data.get("format", {}).get("duration")
        _probe_cache[key] = {"duration_sec": float(dur) if dur else None, "width": video.get("width"),
                             "height": video.get("height"),
                             "has_audio": any(s.get("codec_type") == "audio" for s in data.get("streams", []))}
    return _probe_cache[key]


def _video_path(name: str) -> Path:
    path = (VIDEOS_ROOT / name).resolve()
    if path.parent != VIDEOS_ROOT.resolve() or not path.is_file():
        raise HTTPException(404, f"영상이 없습니다: {name}")
    return path


@app.post("/jobs", response_model=JobState)
def create_job(body: JobCreate):
    """분석 작업을 시작한다. 같은 job_id 는 기존 상태를 돌려준다. 다른 작업이 돌고 있으면 409."""
    with _lock:
        if body.job_id in JOBS:
            return _state(JOBS[body.job_id])
        if any(j.proc.poll() is None for j in JOBS.values()):
            raise HTTPException(409, "다른 분석이 진행 중입니다")
        unknown = [s for s in body.stages if s not in IMPLEMENTED]
        if not body.stages or unknown:
            raise HTTPException(400, f"실행할 수 없는 단계: {unknown or '없음'}")
        video = _video_path(body.video)
        work = OUTPUTS / video.stem
        (work / "work").mkdir(parents=True, exist_ok=True)
        log_path = work / "work" / "job.log"
        cmd = [sys.executable, "-m", "pipeline", str(video), "--work-dir", str(OUTPUTS), "--only", ",".join(body.stages)]
        for stage, values in body.params.items():
            for key, value in values.items():
                cmd += ["--set", f"{stage}.{key}={json.dumps(value, ensure_ascii=False)}"]
        logf = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        logf.close()
        JOBS[body.job_id] = Job(spec=body, proc=proc, log_path=log_path)
        log.info("job %s 시작: %s", body.job_id, " ".join(cmd))
        return _state(JOBS[body.job_id])


@app.get("/jobs/{job_id}", response_model=JobState)
def get_job(job_id: str):
    """작업 상태. 없는 job_id 면 404 (서버를 다시 띄우면 기록이 없어진다)."""
    if job_id not in JOBS:
        raise HTTPException(404, "작업을 찾을 수 없습니다")
    return _state(JOBS[job_id])


@app.post("/jobs/{job_id}/cancel", response_model=JobState)
def cancel_job(job_id: str):
    """작업을 멈춘다."""
    if job_id not in JOBS:
        raise HTTPException(404, "작업을 찾을 수 없습니다")
    job = JOBS[job_id]
    if job.proc.poll() is None:
        job.cancelled = True
        job.proc.terminate()
    return _state(job)


@app.post("/estimate", response_model=EstimateResult)
def estimate(body: EstimateRequest):
    """번역 입력 토큰 (Gemini countTokens, 무료). asr(조건 B 는 roi_ocr 도) 결과가 있어야 한다."""
    from pipeline import translate

    video = _video_path(body.video)
    try:
        return translate.estimate(video, OUTPUTS / video.stem, {"conditions": body.conditions})
    except FileNotFoundError as e:
        raise HTTPException(409, str(e)) from e
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e)) from e


def _state(job: Job) -> JobState:
    code = job.proc.poll()
    if code is not None and job.ended is None:
        job.ended = time.monotonic()
    text = job.log_path.read_text(encoding="utf-8", errors="replace") if job.log_path.exists() else ""
    stages = [m.group(1) for line in text.splitlines() if (m := _STAGE_LINE.match(line))]
    if code is None:
        status, error = "running", None
    elif job.cancelled:
        status, error = "cancelled", None
    elif code == 0:
        status, error = "succeeded", None
    else:
        status, error = "failed", "\n".join(text.strip().splitlines()[-5:]) or f"종료 코드 {code}"
    return JobState(job_id=job.spec.job_id, video=job.spec.video, work_name=Path(job.spec.video).stem,
                    stages=job.spec.stages, status=status, current_stage=stages[-1] if stages else None,
                    elapsed_sec=round((job.ended or time.monotonic()) - job.started, 1), error=error)
