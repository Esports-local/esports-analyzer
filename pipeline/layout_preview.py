"""레이아웃 확인 그림: 영상 한 장면에 OCR 영역을 그리고 읽은 글자를 적는다 (새 대회 좌표 맞출 때).

    python -m pipeline.layout_preview /data/lck_final_1080.mp4 1800 --layout configs/lol/lck_2026.json --out outputs/p.png

- 노랑 상자: rois (roi_ocr 영역). 글자 초록 = 검증 통과, 빨강 = 실패 (괄호 안이 OCR 원문)
- 하늘 상자: screens 규칙 자리. 맨 위 줄에 이 장면의 화면 종류 판별 결과
--layout 을 안 주면 같은 작업 폴더의 header 리그로 고른다 (pipeline/layouts.py).
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pipeline.layouts import load, resolve
from pipeline.media import read_frame_at
from pipeline.roi_ocr import OcrReader, Roi
from pipeline.screen_kinds import classify_frame, load_rules

FONT = "/usr/share/fonts/truetype/nanum/NanumGothicCodingBold.ttf"


def render(frame: np.ndarray, layout: dict, reader) -> Image.Image:
    """장면 한 장에 영역·읽은 글자를 그린 그림."""
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
    work = Path("outputs") / a.video.stem / "work"
    path = resolve(a.layout, work)
    frame = read_frame_at(a.video, a.sec, 1920, 1080)
    if frame is None:
        raise SystemExit(f"{a.sec}초 장면을 읽지 못했습니다")
    render(frame, load(path), OcrReader()).save(a.out)
    print(f"{path} → {a.out}")


if __name__ == "__main__":
    main()
