"""중계 화면 레이아웃 자동 선택 (roi_ocr · segments 의 layout).

레이아웃 파일 하나(configs/lol/<대회>.json)에 rois(roi_ocr 영역)와 screens(segments 화면 종류 규칙, 선택)가 같이 있다.
설정 값이 "auto" 면 work/header.json 의 리그(match.league)로 configs/lol/layouts.json 에서 고른다.
header 가 없거나 op.gg 실패(error)거나 모르는 리그면 default. 경로를 직접 주면 그대로 쓴다.
"""

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

LAYOUTS = Path("configs/lol/layouts.json")
AUTO = "auto"


def header_league(work_dir: Path) -> str | None:
    """work/header.json 의 리그 이름. 없으면 None."""
    path = work_dir / "header.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8")).get("result") or {}
    if result.get("error"):
        return None
    return (result.get("match") or {}).get("league")


def resolve(value: str | None, work_dir: Path, table: Path = LAYOUTS) -> str | None:
    """설정 값을 레이아웃 파일 경로로 바꾼다.

    Args:
        value: 단계 설정 값. "auto" 가 아니면 그대로 돌려준다 (None 포함).
        work_dir: header.json 이 있는 작업 폴더.
    """
    if value != AUTO:
        return value
    conf = json.loads(table.read_text(encoding="utf-8"))
    league = header_league(work_dir)
    chosen = conf["leagues"].get(league) if league else None
    log.info("레이아웃 자동 선택: 리그 %s → %s%s", league, chosen or conf["default"], "" if chosen else " (default)")
    return chosen or conf["default"]


def load(path: str | Path) -> dict:
    """레이아웃 파일 {rois, screens?}."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
