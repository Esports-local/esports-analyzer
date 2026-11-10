"""VOD 분석 파이프라인. 코드에서는 run_pipeline / run_stage 로 실행한다."""

from pipeline.stages import STAGES, run_pipeline, run_stage

__all__ = ["STAGES", "run_pipeline", "run_stage"]
