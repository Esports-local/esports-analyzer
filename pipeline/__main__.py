"""로컬 실행 (컨테이너 안). run_pipeline() 을 감싼 CLI.

docker run --rm -v "$PWD":/app -v ~/vod-data:/data esports-analyzer python -m pipeline /data/game.mp4 --until roi_ocr
docker run --rm -v "$PWD":/app -v ~/vod-data:/data esports-analyzer python -m pipeline /data/game.mp4 --only preprocess,asr
docker run --rm --env-file .env -v "$PWD":/app -v ~/vod-data:/data esports-analyzer python -m pipeline /data/game.mp4 \
    --only translate --set translate.dry_run=true
"""

import argparse
import json
import logging
from pathlib import Path

from pipeline.stages import STAGES, run_pipeline


def main():
    """파이프라인 CLI 진입점.

    --only 로 고른 단계만, 또는 --from 부터 --until 까지 실행한다. 미구현 단계는 건너뛴다.
    결과는 <work-dir>/<영상 파일명>/ 아래에 저장된다.

    예: python -m pipeline /data/game.mp4 --from segments --until roi_ocr
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--from", dest="start", choices=STAGES)
    parser.add_argument("--until", choices=STAGES)
    parser.add_argument("--only", help="쉼표로 구분한 단계 목록 (예: preprocess,asr)")
    parser.add_argument("--set", action="append", default=[], metavar="STAGE.KEY=VALUE",
                        help='단계 설정 덮어쓰기. 값은 JSON 으로 읽고 안 되면 문자열 (예: translate.conditions=["A","B","C"])')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    stages = [s.strip() for s in args.only.split(",") if s.strip()] if args.only else None
    try:
        run_pipeline(args.video, args.work_dir, stages=stages, start=args.start, until=args.until,
                     params=parse_sets(args.set))
    except ValueError as e:
        parser.error(str(e))


def parse_sets(items: list[str]) -> dict[str, dict]:
    """["translate.max_lines=20", ...] → {"translate": {"max_lines": 20}}.

    Raises:
        ValueError: STAGE.KEY=VALUE 형식이 아님.
    """
    params: dict[str, dict] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        stage, dot, name = key.partition(".")
        if not (sep and dot and stage and name):
            raise ValueError(f"--set 은 STAGE.KEY=VALUE 형식이어야 합니다: {item}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        params.setdefault(stage, {})[name] = value
    return params


if __name__ == "__main__":
    main()
