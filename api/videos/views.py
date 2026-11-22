"""/videos 엔드포인트: 고를 수 있는 원본 영상 목록."""

import json
import os
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from api.paths import VIDEOS_ROOT, video_path
from api.videos.schemas import VideoInfo

router = APIRouter()

VIDEO_EXTS = {".mp4", ".mov", ".mkv"}
MEDIA_TYPES = {".mp4": "video/mp4", ".mov": "video/quicktime", ".mkv": "video/x-matroska"}
# 원본 영상을 canvas 로 읽어도 되는 화면 주소 (ui). 중계 영상은 로컬에서만 쓰므로 * 로 열지 않는다
VIDEO_ALLOW_ORIGIN = os.environ.get("VIDEO_ALLOW_ORIGIN", "http://localhost:8000")
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


@router.get("/videos/{name}/file", response_class=FileResponse)
def video_file(name: str):
    """원본 영상 파일 그대로 (브라우저 재생용, ui 의 컷 확인 화면).

    Range 요청을 받으므로 브라우저는 앞쪽 재생 정보(moov)와 보는 위치 주변만 받는다 (faststart mp4 기준).
    ui 가 다른 출처(:8000)에서 프레임을 canvas 로 떠야 해서 그 출처(VIDEO_ALLOW_ORIGIN)만 Access-Control-Allow-Origin 으로 허용한다.

    Raises:
        HTTPException: 404 — VIDEOS_ROOT 바로 아래 영상 파일이 아님.
    """
    try:
        path = video_path(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    if path.suffix.lower() not in VIDEO_EXTS:
        raise HTTPException(404, f"영상이 없습니다: {name}")
    return FileResponse(path, media_type=MEDIA_TYPES[path.suffix.lower()],
                        headers={"Access-Control-Allow-Origin": VIDEO_ALLOW_ORIGIN, "Cache-Control": "private, max-age=3600"})


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
