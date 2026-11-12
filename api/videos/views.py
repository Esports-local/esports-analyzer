"""/videos 엔드포인트: 고를 수 있는 원본 영상 목록."""

import json
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.paths import VIDEOS_ROOT
from api.videos.schemas import VideoInfo

router = APIRouter()

VIDEO_EXTS = {".mp4", ".mov", ".mkv"}
_probe_cache: dict[tuple, dict] = {}


@router.get("/videos", response_model=list[VideoInfo])
def list_videos():
    """VIDEOS_ROOT 의 영상 목록. 1920x1080 이고 소리가 있는 영상만 selectable."""
    if not VIDEOS_ROOT.is_dir():
        raise HTTPException(500, f"영상 폴더가 없습니다: {VIDEOS_ROOT}")
    out = []
    for path in sorted(VIDEOS_ROOT.iterdir()):
        if path.suffix.lower() not in VIDEO_EXTS or path.name.startswith("."):
            continue
        info = probe(path)
        reason = None
        if (info["width"], info["height"]) != (1920, 1080):
            reason = f"{info['width']}x{info['height']} (1920x1080 만 가능)"
        elif not info["has_audio"]:
            reason = "소리 트랙 없음"
        out.append(VideoInfo(name=path.name, size_bytes=path.stat().st_size, selectable=reason is None,
                             reason=reason, **info))
    return out


def probe(path: Path) -> dict:
    """ffprobe 로 길이·해상도·소리 유무. (경로, 크기, 수정 시각)으로 캐시."""
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime)
    if key not in _probe_cache:
        res = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height",
                              "-of", "json", str(path)], capture_output=True, text=True)
        data = json.loads(res.stdout or "{}")
        video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
        dur = data.get("format", {}).get("duration")
        _probe_cache[key] = {"duration_sec": float(dur) if dur else None, "width": video.get("width"),
                             "height": video.get("height"),
                             "has_audio": any(s.get("codec_type") == "audio" for s in data.get("streams", []))}
    return _probe_cache[key]
