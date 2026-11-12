"""공통 엔드포인트 (/health)."""

from fastapi import APIRouter

from pipeline.stages import IMPLEMENTED, STAGES

router = APIRouter()


@router.get("/health")
def health():
    """서버 상태와 단계 목록."""
    return {"status": "ok", "gpu": False, "stages": STAGES, "implemented": list(IMPLEMENTED)}
