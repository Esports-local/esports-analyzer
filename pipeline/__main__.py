"""로컬 실행: uv run python -m pipeline ep01.mp4 --until merge"""

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

    work_dir = args.work_dir / args.video.stem
    work_dir.mkdir(parents=True, exist_ok=True)
    for stage in STAGES[STAGES.index(args.start) : STAGES.index(args.until) + 1]:
        print(f"[{stage}]")
        run_stage(stage, args.video, work_dir)


if __name__ == "__main__":
    main()
