"""숏폼 BGM: assets/bgm/ 의 곡 중 하나를 숏폼 key 로 골라 렌더 때 해설 아래에 깐다.

곡은 사람이 라이선스를 확인하고 넣는다 (YouTube 오디오 보관함 추천, git 에는 올리지 않음).
assets/bgm/bgm.json 이 있으면 그 목록만 쓴다:
    [{"file": "track.mp3", "title": "...", "source": "YouTube Audio Library", "license": "...",
      "attribution": "설명란에 넣을 출처 문구 (없으면 빈 문자열)"}]
bgm.json 이 없으면 폴더의 오디오 파일을 전부 쓴다 (출처 정보 없음). 곡이 없으면 BGM 없이 렌더한다.
같은 key 는 항상 같은 곡 (key 해시로 고름).
"""

import hashlib
import json
import os
from pathlib import Path

BGM_DIR = Path(os.environ.get("BGM_DIR", Path(__file__).resolve().parents[2] / "assets" / "bgm"))
AUDIO_EXT = (".mp3", ".m4a", ".wav", ".ogg", ".flac")


def tracks(root: Path | None = None) -> list[dict]:
    """쓸 수 있는 곡 [{file(절대 경로), title, source, license, attribution}] (파일 이름순)."""
    root = root or BGM_DIR
    listing = root / "bgm.json"
    if listing.exists():
        items = json.loads(listing.read_text(encoding="utf-8"))
        out = [{**t, "file": str(root / t["file"])} for t in items if (root / t.get("file", "")).is_file()]
    else:
        out = [{"file": str(p), "title": p.stem, "source": None, "license": None, "attribution": ""}
               for p in (root.iterdir() if root.is_dir() else []) if p.suffix.lower() in AUDIO_EXT]
    return sorted(out, key=lambda t: Path(t["file"]).name)


def pick(key: str, root: Path | None = None) -> dict | None:
    """key 로 곡 하나. 곡이 없으면 None."""
    items = tracks(root)
    if not items:
        return None
    return items[int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(items)]
