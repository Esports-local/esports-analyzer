"""/estimate 요청·응답 모양 (api ↔ analyzer 연동 스펙, docs/PLAN.md 6장). api 는 이 모양만 믿는다."""

from pydantic import BaseModel


class EstimateRequest(BaseModel):
    video: str
    conditions: list[str] = ["A", "B"]


class EstimateResult(BaseModel):
    model: str
    lines: int
    requests: int
    input_tokens: dict[str, int]
    sampled_lines: int  # 실제로 센 대사 수 (나머지는 비율로 늘림)
