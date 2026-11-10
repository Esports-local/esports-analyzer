"""로컬 실행 (컨테이너 안):
docker run --rm -v "$PWD":/app -v ~/vod-data:/data vod-main python -m pipeline /data/ep01.mp4 --until scenes
"""

import argparse
from pathlib import Path

from pipeline.stages import STAGES, run_stage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--from", dest="start", choices=STAGES, default=STAGES[0])
    parser.add_argument("--until", choices=STAGES, default=STAGES[-1])
    args = parser.parse_args()

    start, until = STAGES.index(args.start), STAGES.index(args.until)
    if start > until:
        parser.error(f"--from {args.start} 이 --until {args.until} 보다 뒤 단계입니다")

    work_dir = args.work_dir / args.video.stem
    work_dir.mkdir(parents=True, exist_ok=True)
    for stage in STAGES[start : until + 1]:
        print(f"[{stage}]")
        run_stage(stage, args.video, work_dir)


if __name__ == "__main__":
    main()
