"""엔드포인트 공통: 원본 영상·결과 폴더 위치."""

import os
from pathlib import Path

from fastapi import HTTPException

VIDEOS_ROOT = Path(os.environ.get("VIDEOS_ROOT", "/data"))
OUTPUTS = Path(os.environ.get("OUTPUTS_ROOT", "outputs"))


def video_path(name: str) -> Path:
    """VIDEOS_ROOT 바로 아래 영상 파일. 없거나 폴더 밖이면 404."""
    path = (VIDEOS_ROOT / name).resolve()
    if path.parent != VIDEOS_ROOT.resolve() or not path.is_file():
        raise HTTPException(404, f"영상이 없습니다: {name}")
    return path
