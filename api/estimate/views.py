"""/estimate 엔드포인트: 번역 예상 입력 토큰."""

from fastapi import APIRouter, HTTPException

from api.estimate.schemas import EstimateRequest, EstimateResult
from api.paths import OUTPUTS, video_path

router = APIRouter()


@router.post("/estimate", response_model=EstimateResult)
def estimate(body: EstimateRequest):
    """번역 입력 토큰 (Gemini countTokens, 무료). asr(조건 B 는 roi_ocr 도) 결과가 있어야 한다."""
    from pipeline import translate

    try:
        video = video_path(body.video)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    try:
        return translate.estimate(video, OUTPUTS / video.stem, {"conditions": body.conditions,
                                                                "text_overrides": body.text_overrides})
    except FileNotFoundError as e:  # 앞 단계 결과가 없음
        raise HTTPException(409, str(e)) from e
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e)) from e
