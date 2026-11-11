"""ffprobe / ffmpeg 공통 함수."""

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pipeline.schema import Segment


def probe(video: Path) -> dict:
    """ffprobe 로 영상의 포맷·스트림 정보를 읽는다.

    Args:
        video: 영상 경로.

    Returns:
        ffprobe JSON 출력 ({"format": ..., "streams": [...]}).

    Raises:
        subprocess.CalledProcessError: 파일이 없거나 영상이 아님.
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(video)],
        check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(out)


def video_stream(info: dict) -> dict:
    """probe 결과에서 첫 번째 영상 스트림을 꺼낸다.

    Args:
        info: probe() 결과.

    Returns:
        영상 스트림 정보 (codec_name, width, height, r_frame_rate 등).

    Raises:
        ValueError: 영상 스트림이 없음.
    """
    for s in info["streams"]:
        if s["codec_type"] == "video":
            return s
    raise ValueError("video stream not found")


def _rate(value: str | None) -> float:
    """ffprobe 의 "30000/1001" 형식 프레임레이트를 float 로 바꾼다. 값이 없거나 "0/0" 이면 0.0."""
    if not value or value in ("0/0", ""):
        return 0.0
    return float(Fraction(value))


def fps_of(stream: dict) -> float:
    """분석에 쓸 프레임레이트를 고른다.

    기본 프레임레이트(r_frame_rate)가 정상 범위(1~60)면 그것을 쓴다. VFR 녹화본은 평균(avg_frame_rate)이
    낮게 나와 시간 해상도가 떨어지기 때문이다. 범위를 벗어나면(타임베이스 값 등) 평균을 쓴다.

    Args:
        stream: video_stream() 결과.

    Returns:
        초당 프레임 수.
    """
    r, avg = _rate(stream.get("r_frame_rate")), _rate(stream.get("avg_frame_rate"))
    return r if 1 <= r <= 60 else avg


def read_frames(video: Path, width: int, height: int, fps: float) -> np.ndarray:
    """영상 전체 프레임을 작은 크기로 읽는다.

    fps 필터로 고정 프레임레이트(CFR)에 맞춰 읽는다. VFR 영상이어도 프레임 i 의 시각이 항상 i / fps 가 된다
    (빠진 자리는 앞 프레임 복제, 겹치면 버림). 전체를 메모리에 올리므로 작은 크기(예: 48x27)로만 쓴다.
    ffmpeg 출력을 bytearray 에 바로 쌓아, 추가 복사 없이 쓰기 가능한 배열로 돌려준다.

    Args:
        video: 영상 경로.
        width: 출력 너비.
        height: 출력 높이.
        fps: 읽을 프레임레이트 (fps_of() 결과).

    Returns:
        (N, height, width, 3) uint8 RGB 배열 (쓰기 가능).

    Raises:
        RuntimeError: ffmpeg 가 실패함.
    """
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(video), "-an",
         "-vf", f"fps={fps!r},scale={width}:{height}",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    buf = bytearray()
    try:
        while chunk := proc.stdout.read(1 << 22):
            buf += chunk
        err = proc.stderr.read().decode(errors="replace")
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 프레임 읽기 실패 ({proc.returncode}): {err.strip()}")
    frame = width * height * 3
    usable = len(buf) - len(buf) % frame
    del buf[usable:]
    return np.frombuffer(buf, np.uint8).reshape(-1, height, width, 3)


def is_vfr(video: Path, fps: float, tolerance: float = 0.5) -> bool:
    """가변 프레임레이트(VFR) 영상인지 판단한다.

    디코딩 없이 영상 패킷 시각만 읽어, 인접 간격이 1/fps 에서 tolerance 비율 이상 벗어나는 곳이 하나라도 있으면
    VFR 로 본다.

    Args:
        video: 영상 경로.
        fps: 기준 프레임레이트.
        tolerance: 허용 오차 (1/fps 대비 비율).

    Returns:
        VFR 이면 True.
    """
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


def segment_at(segments: list["Segment"], t: float) -> "Segment | None":
    """시각 t 가 속한 segment 를 찾는다. 전환 구간이면 None."""
    for s in segments:
        if s.start_sec <= t < s.end_sec:
            return s
    return None


def frame_size(video: Path, height: int | None) -> tuple[int, int]:
    """프레임을 읽을 크기를 정한다.

    Args:
        video: 영상 경로.
        height: 원하는 높이. None 이면 원본 크기.

    Returns:
        (너비, 높이). 비율을 유지하고 너비는 짝수로 맞춘다.
    """
    v = video_stream(probe(video))
    w, h = int(v["width"]), int(v["height"])
    if height is None or height == h:
        return (w, h)
    return (round(w * height / h / 2) * 2, height)


def read_frame_at(video: Path, t: float, width: int, height: int) -> np.ndarray | None:
    """시각 t 의 프레임 한 장을 읽는다.

    Args:
        video: 영상 경로.
        t: 시각(초).
        width: 출력 너비.
        height: 출력 높이.

    Returns:
        (height, width, 3) uint8 RGB. 영상 끝을 넘으면 None.
    """
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
                          "-vf", f"scale={width}:{height}", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:"],
                         check=True, capture_output=True).stdout
    if len(raw) != width * height * 3:
        return None
    return np.frombuffer(raw, np.uint8).reshape(height, width, 3)


def sample_segment_frames(video: Path, segments: list["Segment"], fps: float, height: int | None):
    """segment 구간에서 프레임을 뽑는다.

    전체 영상을 fps 간격(CFR)으로 읽어 segment 안에 있는 프레임만 낸다 (전환 구간은 건너뜀).
    그 간격보다 짧아 프레임을 하나도 받지 못한 segment 는 가운데 프레임을 따로 뽑는다.
    ffmpeg 가 중간에 실패하면 일부만 처리한 결과를 저장하지 않도록 예외를 낸다.
    호출 쪽이 중간에 멈추거나 예외가 나면 ffmpeg 프로세스를 정리한다.

    Args:
        video: 영상 경로.
        segments: segments 단계 결과.
        fps: 초당 뽑을 프레임 수.
        height: 프레임 높이. None 이면 원본.

    Yields:
        (segment, 시각(초), BGR 프레임 (H, W, 3) uint8).

    Raises:
        RuntimeError: ffmpeg 가 실패함.
    """
    w, h = frame_size(video, height)
    size = w * h * 3
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(video), "-an",
                             "-vf", f"fps={fps},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    covered: set[int] = set()
    try:
        k = 0
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            t = k / fps
            k += 1
            seg = segment_at(segments, t)
            if seg is not None:
                covered.add(seg.id)
                yield seg, t, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        err = proc.stderr.read().decode(errors="replace")
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg 프레임 읽기 실패 ({proc.returncode}): {err.strip()}")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    for seg in segments:  # 샘플이 하나도 없던 짧은 segment 는 가운데 프레임
        if seg.id in covered:
            continue
        t = (seg.start_sec + seg.end_sec) / 2
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
                              "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:"],
                             check=True, capture_output=True).stdout
        if len(raw) == size:
            yield seg, t, np.frombuffer(raw, np.uint8).reshape(h, w, 3)
