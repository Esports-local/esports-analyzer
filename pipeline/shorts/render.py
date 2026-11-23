"""숏폼 렌더: 원본 영상의 구간들을 세로(9:16) 영상 하나로 만든다 (ffmpeg 한 번).

화면: 1920x1080 중계 화면 가운데에서 9:16(608x1080)을 잘라 720x1280 으로 키운다. 자막·제목은 넣지 않는다.

구간이 여러 개면 순서대로 이어 붙인다.
결과는 임시 파일에 쓴 뒤 이름을 바꿔, 중간에 실패해도 반쯤 쓴 파일이 결과로 보이지 않게 한다.
"""

import json
import logging
import subprocess
import time
from pathlib import Path

log = logging.getLogger(__name__)

W, H = 720, 1280
CROP = (656, 0, 608, 1080)        # x, y, w, h (1920x1080 기준): 가운데 9:16


def render(video: Path, out_path: Path, segments: list[dict], crf: int = 23, preset: str = "veryfast") -> dict:
    """구간들을 세로 숏폼 mp4 하나로 렌더한다.

    Args:
        video: 원본 영상 (1920x1080).
        out_path: 결과 mp4 경로. 옆에 같은 이름의 .json(렌더 기록)도 쓴다.
        segments: [{start_sec, end_sec}] 원본 기준, 시간순·겹치지 않게.
        crf, preset: libx264 설정.

    Returns:
        렌더 기록 {duration_sec, render_sec, size_bytes, segments}.

    Raises:
        ValueError: 구간이 비었거나 순서가 틀림. (길이·개수 제한은 없다, 11-23)
        RuntimeError: ffmpeg 실패 (마지막 로그 몇 줄 포함).
    """
    segs = _check_segments(segments)
    total = sum(e - s for s, e in segs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    part = out_path.with_name(f".{out_path.stem}.part.mp4")  # 같은 폴더에 써야 이름 바꾸기가 한 번에 된다
    res = subprocess.run(_ffmpeg_cmd(video, segs, part, crf, preset), capture_output=True, text=True)
    if res.returncode != 0:
        part.unlink(missing_ok=True)
        tail = "\n".join(res.stderr.strip().splitlines()[-5:])
        raise RuntimeError(f"ffmpeg 실패 (코드 {res.returncode}): {tail}")
    part.replace(out_path)
    meta = {"duration_sec": round(total, 2), "render_sec": round(time.monotonic() - t0, 2),
            "size_bytes": out_path.stat().st_size, "segments": [{"start_sec": s, "end_sec": e} for s, e in segs]}
    out_path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("숏폼 %s: %.1f초 영상, 렌더 %.1f초", out_path.name, total, meta["render_sec"])
    return meta


def _check_segments(segments: list[dict]) -> list[tuple[float, float]]:
    segs = [(float(s["start_sec"]), float(s["end_sec"])) for s in segments]
    if not segs:
        raise ValueError("구간이 없습니다")
    for (s, e), nxt in zip(segs, segs[1:] + [None]):
        if s < 0 or e <= s:
            raise ValueError(f"잘못된 구간: {s}~{e}")
        if nxt and nxt[0] < e:
            raise ValueError("구간은 시간순이고 겹치지 않아야 합니다")
    return segs


def _ffmpeg_cmd(video: Path, segs: list[tuple[float, float]], out: Path, crf: int, preset: str) -> list[str]:
    """구간마다 입력을 하나씩 열고(-ss/-to 로 빠르게 찾기), 가운데 9:16 을 잘라 키운 뒤 concat."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for s, e in segs:
        cmd += ["-ss", f"{s:.3f}", "-to", f"{e:.3f}", "-i", str(video)]
    cx, cy, cw, ch = CROP
    parts, concat_in = [], ""
    for i in range(len(segs)):
        parts.append(
            f"[{i}:v]scale=1920:1080,setsar=1,crop={cw}:{ch}:{cx}:{cy},scale={W}:{H},fps=30,format=yuv420p[v{i}];"
            f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo[a{i}]"
        )
        concat_in += f"[v{i}][a{i}]"
    graph = ";".join(parts) + f";{concat_in}concat=n={len(segs)}:v=1:a=1[vout][ca]"
    cmd += ["-filter_complex", graph, "-map", "[vout]", "-map", "[ca]",
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart", str(out)]
    return cmd
