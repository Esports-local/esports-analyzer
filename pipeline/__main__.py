"""로컬 실행 (컨테이너 안). run_pipeline() 을 감싼 CLI.

docker run --rm -v "$PWD":/app -v ~/vod-data:/data esports-analyzer python -m pipeline /data/game.mp4 --until roi_ocr
docker run --rm -v "$PWD":/app -v ~/vod-data:/data esports-analyzer python -m pipeline /data/game.mp4 --only preprocess,asr
docker run --rm -v "$PWD":/app -v ~/vod-data:/data esports-analyzer python -m pipeline /data/game.mp4 --only roi_ocr \
    --match data/matches/261108_T1_BLG_g2.json
"""

import argparse
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
    parser.add_argument("--match", type=Path, help="경기 정보 match.json (python -m pipeline.match 로 받음)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    stages = [s.strip() for s in args.only.split(",") if s.strip()] if args.only else None
    try:
        params = {"roi_ocr": {"match": str(args.match)}} if args.match else None
        run_pipeline(args.video, args.work_dir, stages=stages, start=args.start, until=args.until, params=params)
    except ValueError as e:
        parser.error(str(e))


if __name__ == "__main__":
    main()
