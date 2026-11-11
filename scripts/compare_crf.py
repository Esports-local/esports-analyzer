"""hls 화질(CRF) 비교: 영상 일부를 CRF 별로 인코딩해 용량·비트레이트·SSIM 을 나란히 본다.

사용 (컨테이너 안):
    python scripts/compare_crf.py /data/ep01.mp4 --start 600 --duration 120 --crf 20 23 26

결과: outputs/crf_compare/<영상명>/crf<N>.mp4 와 표 출력. 눈으로 볼 때는 어두운 장면·그라데이션 위주로 비교한다.
1시간 용량은 구간 비트레이트로 추정한 값이다.
"""

import argparse
import re
import subprocess
from pathlib import Path


def encode(src: Path, start: float, duration: float, crf: int, preset: str, out: Path) -> None:
    """hls 단계와 같은 설정(libx264, yuv420p)으로 구간을 인코딩한다 (오디오 제외)."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(duration), "-i", str(src),
                    "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p", str(out)],
                   check=True)


def ssim(src: Path, start: float, duration: float, enc: Path) -> float | None:
    """원본 같은 구간과의 SSIM (1 에 가까울수록 원본과 비슷). 계산 실패 시 None."""
    r = subprocess.run(["ffmpeg", "-v", "info", "-ss", str(start), "-t", str(duration), "-i", str(src), "-i", str(enc),
                        "-lavfi", "[0:v][1:v]scale2ref[a][b];[a][b]ssim", "-f", "null", "-"],
                       capture_output=True, text=True)
    m = re.search(r"All:([0-9.]+)", r.stderr)
    return float(m.group(1)) if m else None


def main():
    """CRF 값마다 인코딩하고 용량·비트레이트·1시간 추정 용량·SSIM 을 출력한다."""
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--crf", type=int, nargs="+", default=[20, 23, 26])
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--out", type=Path, default=Path("outputs/crf_compare"))
    args = ap.parse_args()

    out_dir = args.out / args.video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{'CRF':>4} {'용량(MB)':>9} {'Mbps':>6} {'1시간(GB)':>9} {'SSIM':>7}")
    for crf in args.crf:
        enc = out_dir / f"crf{crf}.mp4"
        encode(args.video, args.start, args.duration, crf, args.preset, enc)
        size = enc.stat().st_size
        mbps = size * 8 / args.duration / 1e6
        s = ssim(args.video, args.start, args.duration, enc)
        print(f"{crf:>4} {size / 1e6:>9.1f} {mbps:>6.2f} {mbps * 3600 / 8 / 1000:>9.2f} {s if s is not None else '-':>7}")


if __name__ == "__main__":
    main()
