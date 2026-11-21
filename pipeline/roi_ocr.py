"""roi_ocr 단계: 중계 화면의 고정 영역(ROI)을 OCR 로 읽어, 값이 이어진 구간 목록을 만든다.

설정 파일(configs/<종목>/<레이아웃>.json)의 rois 가 영역 목록이다 (screens 는 segments 가 쓴다). 항목마다
- tag, slot, kind: 결과 이름 (예: player_name / blue_top / player)
- ocr_roi [x1, y1, x2, y2]: 기준 해상도(base_resolution, 기본 1920x1080) 픽셀 좌표
- ocr_scale: 작은 글자를 키워서 읽는 배율
- char_map: 글꼴 때문에 잘못 읽는 문자 바꾸기 (예: LCK 숫자 0 → {"o": "0", "O": "0"}). allowed_text 보다 먼저
- allowed_text: 남길 문자 (그 밖의 문자는 지운다)
- validation: regex(전체 일치), min_confidence. 통과하지 못하면 그 샘플은 "못 읽음"
- static: 경기 중 바뀌지 않는 값(선수 이름, 팀 등). 같은 값이 static_lock_sec 동안 이어지면 확정하고
  그 뒤로는 읽지 않는다. 결과는 확정한 구간 하나 (영상 끝까지)

1. 영상을 sample_fps 간격으로 한 장씩 읽어 기준 해상도로 맞춘다 (전체를 메모리에 올리지 않음).
2. 영역마다 앞서 OCR 한 때와 화면이 거의 같으면(change_threshold 미만) 그 값을 다시 쓰고, 달라졌으면 OCR 한다.
3. 같은 값이 이어지면 한 구간으로 합친다. 못 읽은 샘플이 hold_sec 이하로 끼어 있으면 구간을 끊지 않는다
   (화면 효과로 잠깐 가려지는 경우). 그보다 길면 구간을 닫는다.
"""

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.io import save_stage
from pipeline.layouts import resolve
from pipeline.schema import OcrRun, RoiOcrResult

log = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "layout": "auto",  # 설정 파일 경로, auto 면 header 리그로 고름 (pipeline/layouts.py, configs/lol/layouts.json)
    "base_resolution": [1920, 1080],
    "sample_fps": 2.0,
    "change_threshold": 3.0,  # 영역 회색조 평균 차이(0~255)가 이 값 이상이면 다시 OCR
    "hold_sec": 1.0,          # 못 읽은 샘플이 이 시간 이하로 끼면 구간을 이어 간다
    "static_lock_sec": 5.0,   # static 영역: 같은 값이 이 시간 이어지면 확정
    "crop": None,             # [x, y, w, h] 원본 픽셀. 화면 녹화본처럼 중계 화면이 일부일 때 그 부분만 잘라 쓴다
}


def run(video: Path, work_dir: Path, params: dict | None = None) -> Path:
    """roi_ocr 단계를 실행한다.

    Args:
        video: 원본 영상 경로.
        work_dir: 작업 디렉터리.
        params: DEFAULT_PARAMS 를 덮어쓸 설정 (layout, sample_fps …).

    Returns:
        work/roi_ocr.json 경로.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    p["layout"] = resolve(p["layout"], work_dir)
    layout = Path(p["layout"])
    rois = [Roi.from_config(c, i) for i, c in enumerate(json.loads(layout.read_text(encoding="utf-8"))["rois"])]
    width, height = p["base_resolution"]
    reader = OcrReader()
    step = 1 / p["sample_fps"]

    frames = calls = 0
    for i, frame in enumerate(iter_frames(video, width, height, p["sample_fps"], p["crop"])):
        t = i * step
        frames += 1
        gray = np.asarray(Image.fromarray(frame).convert("L"), np.int16)
        for roi in rois:
            if roi.locked:
                continue
            x1, y1, x2, y2 = roi.box
            crop = gray[y1:y2, x1:x2]
            if roi.last_crop is None or np.abs(crop - roi.last_crop).mean() >= p["change_threshold"]:
                roi.last_crop = crop
                roi.last_read = roi.read(frame, reader)
                calls += 1
            roi.add(t, step, roi.last_read, p["hold_sec"])
            if roi.static and roi.try_lock(p["static_lock_sec"]):
                log.info("roi_ocr: %s/%s 확정 %r (%.1f초)", roi.tag, roi.slot, roi.last_read.text, t)
    for roi in rois:
        if roi.locked:
            roi.extend_to(frames * step)
    runs = sorted((r for roi in rois for r in roi.finish()), key=lambda r: (r.start_sec, r.tag, r.slot))
    log.info("roi_ocr: 프레임 %d장, 영역 %d개, OCR %d회 (재사용 %d회), 구간 %d개",
             frames, len(rois), calls, frames * len(rois) - calls, len(runs))

    result = RoiOcrResult(layout=str(layout), sample_fps=p["sample_fps"], base_resolution=(width, height),
                          rois=len(rois), frames=frames, ocr_calls=calls, runs=runs)
    return save_stage(work_dir, "roi_ocr", result, p)


def iter_frames(video: Path, width: int, height: int, fps: float, crop: list[int] | None = None):
    """영상을 fps 간격으로 한 장씩 읽는다 (CFR, width x height RGB).

    crop [x, y, w, h] 가 있으면 원본에서 그 부분을 먼저 잘라 낸 뒤 크기를 맞춘다.

    Yields:
        (height, width, 3) uint8 배열. 프레임 i 의 시각은 i / fps.

    Raises:
        RuntimeError: ffmpeg 가 실패함.
    """
    vf = f"fps={fps!r},scale={width}:{height}"
    if crop:
        x, y, w, h = crop
        vf = f"crop={w}:{h}:{x}:{y}," + vf
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(video), "-an", "-vf", vf,
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    size = width * height * 3
    try:
        while len(buf := proc.stdout.read(size)) == size:
            yield np.frombuffer(buf, np.uint8).reshape(height, width, 3)
        err = proc.stderr.read().decode(errors="replace")
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 프레임 읽기 실패 ({proc.returncode}): {err.strip()}")


class OcrReader:
    """RapidOCR 글자 인식(rec)만 쓴다. 영역이 정해져 있으므로 글자 검출(det)·방향 분류(cls)는 끈다."""

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR  # 무거운 모델 로딩은 단계 실행 때만

        self.engine = RapidOCR()

    def __call__(self, image: np.ndarray) -> tuple[str, float]:
        """이미지 한 장에서 글자 한 줄을 읽는다. 못 읽으면 ("", 0.0)."""
        res, _ = self.engine(image, use_det=False, use_cls=False, use_rec=True)
        if not res:
            return "", 0.0
        text, conf = res[0]
        return text.strip(), float(conf)


@dataclass
class Reading:
    raw: str
    text: str
    conf: float
    valid: bool


@dataclass
class Roi:
    tag: str
    slot: str
    kind: str | None
    box: tuple[int, int, int, int]
    scale: int = 1
    char_map: dict[str, str] = field(default_factory=dict)
    allowed: str | None = None
    regex: str | None = None
    min_conf: float = 0.0
    static: bool = False
    locked: bool = False
    last_crop: np.ndarray | None = None
    last_read: Reading | None = None
    runs: list[OcrRun] = field(default_factory=list)
    _cur: dict | None = None  # 진행 중인 구간 {text, raw, start, end, confs}

    @classmethod
    def from_config(cls, c: dict, index: int) -> "Roi":
        """설정 항목 하나를 읽는다. 지금은 ocr_roi 가 있는 항목만 쓴다 (게이트 템플릿은 쓰지 않음).

        Raises:
            ValueError: ocr_roi 가 없거나 좌표가 잘못됨.
        """
        box = c.get("ocr_roi")
        if not box or len(box) != 4 or box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError(f"설정 {index}번({c.get('tag')}): ocr_roi 는 [x1, y1, x2, y2] 여야 합니다: {box}")
        v = c.get("validation", {})
        return cls(tag=c["tag"], slot=c.get("slot", ""), kind=c.get("kind"), box=tuple(box),
                   scale=int(c.get("ocr_scale", 1)), char_map=dict(c.get("char_map") or {}), allowed=c.get("allowed_text"), regex=v.get("regex"),
                   min_conf=float(v.get("min_confidence", 0.0)),
                   static=bool(c.get("static", False)))

    def read(self, frame: np.ndarray, reader: OcrReader) -> Reading:
        """영역을 잘라 OCR 하고 정리·검증한다."""
        x1, y1, x2, y2 = self.box
        crop = Image.fromarray(frame[y1:y2, x1:x2])
        if self.scale > 1:
            crop = crop.resize((crop.width * self.scale, crop.height * self.scale), Image.LANCZOS)
        raw, conf = reader(np.asarray(crop))
        text = "".join(self.char_map.get(ch, ch) for ch in raw)
        if self.allowed is not None:
            text = "".join(ch for ch in text if ch in self.allowed)
        valid = bool(text) and conf >= self.min_conf and (self.regex is None or re.fullmatch(self.regex, text))
        return Reading(raw=raw, text=text, conf=conf, valid=bool(valid))

    def add(self, t: float, step: float, r: Reading, hold_sec: float) -> None:
        """샘플 하나를 구간에 반영한다."""
        cur = self._cur
        if r.valid:
            if cur and cur["text"] == r.text and t - cur["end"] <= hold_sec + 1e-6:
                cur["end"] = t + step
                cur["confs"].append(r.conf)
                return
            self._close()
            self._cur = {"text": r.text, "raw": r.raw, "start": t, "end": t + step, "confs": [r.conf]}
        elif cur and t + step - cur["end"] > hold_sec + 1e-6:
            self._close()

    def try_lock(self, lock_sec: float) -> bool:
        """static 영역: 진행 중인 구간이 lock_sec 이상이면 확정한다. 앞서 생긴 다른 구간은 버린다.

        Returns:
            이번에 확정했으면 True.
        """
        cur = self._cur
        if self.locked or not cur or cur["end"] - cur["start"] < lock_sec - 1e-6:
            return False
        self.locked = True
        self.runs.clear()
        return True

    def extend_to(self, end_sec: float) -> None:
        """확정한 구간을 end_sec(영상 끝)까지 늘린다."""
        if self._cur:
            self._cur["end"] = max(self._cur["end"], end_sec)

    def finish(self) -> list[OcrRun]:
        """남은 구간을 닫고 전체 구간 목록을 돌려준다."""
        self._close()
        return self.runs

    def _close(self) -> None:
        cur, self._cur = self._cur, None
        if cur:
            self.runs.append(OcrRun(tag=self.tag, slot=self.slot, kind=self.kind, text=cur["text"], raw=cur["raw"],
                                    start_sec=round(cur["start"], 3), end_sec=round(cur["end"], 3),
                                    conf=round(sum(cur["confs"]) / len(cur["confs"]), 3),
                                    samples=len(cur["confs"])))

