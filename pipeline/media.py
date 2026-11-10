"""ffprobe / ffmpeg 공통 함수."""

import json
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np


def probe(video: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(video)],
        check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(out)


def video_stream(info: dict) -> dict:
    for s in info["streams"]:
        if s["codec_type"] == "video":
            return s
    raise ValueError("video stream not found")


def fps_of(stream: dict) -> float:
    rate = stream.get("avg_frame_rate") or stream["r_frame_rate"]
    if rate in ("0/0", ""):
        rate = stream["r_frame_rate"]
    return float(Fraction(rate))


def read_frames(video: Path, width: int, height: int) -> np.ndarray:
    """전체 프레임을 (N, height, width, 3) uint8 RGB 로 읽는다."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-an",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "pipe:"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)
