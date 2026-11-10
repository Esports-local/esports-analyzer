from fastapi import FastAPI, HTTPException

from app.schemas import JobCreate, JobState
from pipeline.stages import STAGES

app = FastAPI(title="VOD main")

# TODO: 큐(Redis 등)와 워커로 교체. 지금은 접수만 하는 인메모리 저장소.
JOBS: dict[str, JobState] = {}


@app.get("/health")
def health():
    return {"status": "ok", "gpu": False, "stages": STAGES}


@app.post("/jobs", response_model=JobState)
def create_job(body: JobCreate):
    if body.job_id in JOBS:
        return JOBS[body.job_id]
    state = JobState(job_id=body.job_id, status="queued")
    JOBS[body.job_id] = state
    return state


@app.get("/jobs/{job_id}", response_model=JobState)
def get_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "job not found")
    return JOBS[job_id]


@app.post("/jobs/{job_id}/cancel", response_model=JobState)
def cancel_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "job not found")
    JOBS[job_id].status = "cancelled"
    return JOBS[job_id]
