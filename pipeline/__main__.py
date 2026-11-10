"""로컬 실행 (컨테이너 안). run_pipeline() 을 감싼 CLI.

docker run --rm -v "$PWD":/app -v ~/vod-data:/data vod-main python -m pipeline /data/ep01.mp4 --until faces
docker run --rm -v "$PWD":/app -v ~/vod-data:/data vod-main python -m pipeline /data/ep01.mp4 --only segments,faces
"""

import argparse
import logging
from pathlib import Path

from pipeline.stages import STAGES, run_pipeline


def main():
    """파이프라인 CLI 진입점.

    --only 로 고른 단계만, 또는 --from 부터 --until 까지 실행한다. 미구현 단계는 건너뛴다.
    결과는 <work-dir>/<영상 파일명>/ 아래에 저장된다.

    예: python -m pipeline /data/ep01.mp4 --from segments --until faces
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--from", dest="start", choices=STAGES)
    parser.add_argument("--until", choices=STAGES)
    parser.add_argument("--only", help="쉼표로 구분한 단계 목록 (예: segments,faces)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    stages = [s.strip() for s in args.only.split(",") if s.strip()] if args.only else None
    try:
        run_pipeline(args.video, args.work_dir, stages=stages, start=args.start, until=args.until)
    except ValueError as e:
        parser.error(str(e))


if __name__ == "__main__":
    main()
