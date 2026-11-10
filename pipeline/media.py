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


def _rate(value: str | None) -> float:
    if not value or value in ("0/0", ""):
        return 0.0
    return float(Fraction(value))


def fps_of(stream: dict) -> float:
    """분석에 쓸 프레임레이트.

    기본 프레임레이트(r_frame_rate)가 정상 범위(1~60)면 그것을 쓴다. VFR 녹화본은 평균(avg_frame_rate)이
    낮게 나와 시간 해상도가 떨어지기 때문. 범위를 벗어나면(타임베이스 값 등) 평균을 쓴다.
    """
    r, avg = _rate(stream.get("r_frame_rate")), _rate(stream.get("avg_frame_rate"))
    return r if 1 <= r <= 60 else avg


def read_frames(video: Path, width: int, height: int, fps: float) -> np.ndarray:
    """전체 프레임을 (N, height, width, 3) uint8 RGB 로 읽는다.

    fps 필터로 고정 프레임레이트(CFR)에 맞춰 읽는다. VFR 영상이어도 프레임 i 의 시각이 항상 i / fps 가 된다
    (빠진 자리는 앞 프레임 복제, 겹치면 버림).
    """
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-an",
         "-vf", f"fps={fps!r},scale={width}:{height}",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)


def is_vfr(video: Path, fps: float, tolerance: float = 0.5) -> bool:
    """영상 패킷 시각 간격이 1/fps 에서 tolerance 비율 이상 벗어나는 경우가 있으면 VFR 로 본다."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
         "-of", "csv=p=0", str(video)],
        check=True, capture_output=True, text=True,
    ).stdout
    pts = np.sort(np.array([float(x) for x in out.split() if x not in ("", "N/A")]))
    if len(pts) < 3:
        return False
    gaps = np.diff(pts)
    return bool(np.any(np.abs(gaps - 1 / fps) > tolerance / fps))
