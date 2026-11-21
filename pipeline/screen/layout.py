"""중계 화면 레이아웃: 자동 선택 (roi_ocr · segments 의 layout) + 확인 그림.

레이아웃 파일 하나(configs/lol/<대회>.json)에 rois(roi_ocr 영역)와 screens(segments 화면 종류 규칙, 선택)가 같이 있다.
설정 값이 "auto" 면 work/header.json 의 리그(match.league)로 configs/lol/layouts.json 에서 고른다.
header 가 없거나 op.gg 실패(error)거나 모르는 리그면 default. 경로를 직접 주면 그대로 쓴다.

확인 그림 (새 대회 좌표 맞출 때): 영상 한 장면에 영역을 그리고 읽은 글자·화면 종류를 적는다.
    python -m pipeline.screen.layout /data/lck_final_1080.mp4 1800 --layout configs/lol/lck_2026.json --out outputs/p.png
- 노랑 상자: rois (roi_ocr 영역). 글자 초록 = 검증 통과, 빨강 = 실패 (괄호 안이 OCR 원문)
- 하늘 상자: screens 규칙 자리. 맨 아래 줄에 이 장면의 화면 종류 판별 결과
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from pipeline.io import stage_path

log = logging.getLogger(__name__)

LAYOUTS = Path("configs/lol/layouts.json")
AUTO = "auto"


def header_league(work_dir: Path) -> str | None:
    """work/header.json 의 리그 이름. 없으면 None."""
    path = stage_path(work_dir, "header")
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


FONT = "/usr/share/fonts/truetype/nanum/NanumGothicCodingBold.ttf"


def render(frame: np.ndarray, layout: dict, reader):
    """장면 한 장에 영역·읽은 글자를 그린 그림 (PIL Image)."""
    from PIL import Image, ImageDraw, ImageFont

    from pipeline.screen.roi_ocr import Roi
    from pipeline.screen.rules import classify_frame, load_rules

    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(FONT, 14)
        big = ImageFont.truetype(FONT, 28)
    except OSError:
        font = big = ImageFont.load_default()
    for i, c in enumerate(layout.get("rois", [])):
        roi = Roi.from_config(c, i)
        r = roi.read(frame, reader)
        x1, y1, x2, y2 = roi.box
        draw.rectangle(roi.box, outline=(255, 220, 0), width=2)
        label = r.text if r.valid else f"{r.text}({r.raw})"
        ty = y2 + 1 if y2 + 16 < img.height else y1 - 16
        draw.text((x1, ty), label, fill=(0, 255, 0) if r.valid else (255, 60, 60), font=font,
                  stroke_width=2, stroke_fill=(0, 0, 0))
    screens = layout.get("screens")
    if screens:
        rules, default, _ = load_rules(screens)
        for rule in rules:
            draw.rectangle(rule.spot.roi, outline=(0, 200, 255), width=2)
            for spot in rule.read.values():
                draw.rectangle(spot.roi, outline=(0, 200, 255), width=1)
        kind = classify_frame(frame, rules, default, reader)
        draw.text((10, img.height - 40), f"화면 종류: {kind}", fill=(0, 200, 255), font=big,
                  stroke_width=3, stroke_fill=(0, 0, 0))
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description="레이아웃 확인 그림")
    ap.add_argument("video", type=Path)
    ap.add_argument("sec", type=float, help="장면 시각(초)")
    ap.add_argument("--layout", default="auto", help="레이아웃 파일 (기본 auto: header 리그)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from pipeline.media import read_frame_at
    from pipeline.screen.roi_ocr import OcrReader

    work = Path("outputs") / a.video.stem / "work"
    path = resolve(a.layout, work)
    frame = read_frame_at(a.video, a.sec, 1920, 1080)
    if frame is None:
        raise SystemExit(f"{a.sec}초 장면을 읽지 못했습니다")
    render(frame, load(path), OcrReader()).save(a.out)
    print(f"{path} → {a.out}")


if __name__ == "__main__":
    main()
