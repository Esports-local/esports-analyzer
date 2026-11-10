from fastapi import FastAPI, HTTPException

from app.schemas import JobCreate, JobState
from pipeline.stages import STAGES

app = FastAPI(title="VOD main")

# TODO: 큐(Redis 등)와 워커로 교체. 지금은 접수만 하는 인메모리 저장소.
JOBS: dict[str, JobState] = {}


@app.get("/health")
def health():
    """서버 상태와 파이프라인 단계 목록을 돌려준다."""
    return {"status": "ok", "gpu": False, "stages": STAGES}


@app.post("/jobs", response_model=JobState)
def create_job(body: JobCreate):
    """분석 작업을 접수한다 (임시: 메모리에 저장만 하고 실행하지 않음).

    같은 job_id 가 이미 있으면 새로 만들지 않고 기존 상태를 돌려준다.
    """
    if body.job_id in JOBS:
        return JOBS[body.job_id]
    state = JobState(job_id=body.job_id, status="queued")
    JOBS[body.job_id] = state
    return state


@app.get("/jobs/{job_id}", response_model=JobState)
def get_job(job_id: str):
    """작업 상태를 조회한다. 없는 job_id 면 404."""
    if job_id not in JOBS:
        raise HTTPException(404, "job not found")
    return JOBS[job_id]


@app.post("/jobs/{job_id}/cancel", response_model=JobState)
def cancel_job(job_id: str):
    """작업을 취소 상태로 바꾼다. 없는 job_id 면 404."""
    if job_id not in JOBS:
        raise HTTPException(404, "job not found")
    JOBS[job_id].status = "cancelled"
    return JOBS[job_id]
