"""segments 단계의 화면 종류 판별: live(경기) · replay(하이라이트) · postgame(경기 뒤 통계) · break(그 밖).

TransNetV2 는 장면이 바뀌는 지점만 찾고 장면이 무엇인지는 모른다. 그래서 영상을 sample_sec 간격으로 한 장씩 읽어
화면마다 정해진 자리의 글자를 RapidOCR(글자 인식만)로 읽고, 레이아웃 파일(configs/<종목>/<대회>.json) screens 의
규칙을 위에서부터 맞춰 처음 맞는 종류로 정한다. 새 모델은 없다.
- contains: 그 자리 글자에 이 말(하나 또는 목록 중 하나)이 들어 있으면 (대문자·영숫자만 남겨 비교, 조금 틀려도 됨)
- regex: 그 자리 글자가 이 모양이면 (예: 경기 시간 12:34)
- value: 맞은 글자를 값으로 남긴다 (live 의 game_time → 경기 몇 초인지)
- read: 맞았을 때 더 읽을 자리 (postgame 의 경기 길이 · 승패)

segment(장면 전환 사이)의 종류는 그 안 화면들의 다수결. 경기 구간(game)은 live 화면의 경기 시간과 header 의 경기 길이로 정하고,
postgame 화면에서 읽은 경기 길이·승패로 맞는지 확인한다.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np
from PIL import Image

CONTAINS_MIN_RATIO = 0.8


@dataclass
class Spot:
    """화면의 글자 자리 하나."""

    roi: tuple[int, int, int, int]
    scale: int = 1
    allowed: str | None = None
    char_map: dict[str, str] = field(default_factory=dict)
    regex: str | None = None
    contains: list[str] | None = None  # 이 말 중 하나라도 들어 있으면

    @classmethod
    def from_config(cls, c: dict) -> "Spot":
        x1, y1, x2, y2 = c["roi"]
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"roi 는 [x1, y1, x2, y2] 여야 합니다: {c['roi']}")
        return cls(roi=(x1, y1, x2, y2), scale=int(c.get("scale", 1)), allowed=c.get("allowed"),
                   char_map=dict(c.get("char_map") or {}), regex=c.get("regex"), contains=_words(c.get("contains")))

    def read(self, frame: np.ndarray, reader) -> str:
        """자리를 잘라 읽고 char_map · allowed 로 정리한 글자."""
        x1, y1, x2, y2 = self.roi
        crop = Image.fromarray(frame[y1:y2, x1:x2])
        if self.scale > 1:
            crop = crop.resize((crop.width * self.scale, crop.height * self.scale), Image.LANCZOS)
        raw, _ = reader(np.asarray(crop))
        text = "".join(self.char_map.get(ch, ch) for ch in raw)
        if self.allowed is not None:
            text = "".join(ch for ch in text if ch in self.allowed)
        return text

    def matches(self, text: str) -> bool:
        if not text:
            return False
        if self.regex and not re.fullmatch(self.regex, text):
            return False
        if self.contains:
            return any(_contains(text, w) for w in self.contains)
        return True


@dataclass
class Rule:
    kind: str
    spot: Spot
    value: str | None = None
    read: dict[str, Spot] = field(default_factory=dict)


def _words(v) -> list[str] | None:
    return [v] if isinstance(v, str) else (list(v) if v else None)


def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def _contains(text: str, word: str) -> bool:
    """text 안에 word 가 있는지. OCR 이 한두 글자 틀려도 맞다고 본다."""
    t, w = _norm(text), _norm(word)
    if not w or not t:
        return False
    if w in t:
        return True
    n = len(w)
    windows = [t[i:i + n] for i in range(max(1, len(t) - n + 1))]
    return max(SequenceMatcher(None, x, w).ratio() for x in windows) >= CONTAINS_MIN_RATIO


def load_rules(config: dict) -> tuple[list[Rule], str, float]:
    """규칙 파일 내용 → (규칙 목록, 기본 종류, 샘플 간격 초)."""
    rules = [Rule(kind=r["kind"], spot=Spot.from_config(r), value=r.get("value"),
                  read={k: Spot.from_config(v) for k, v in (r.get("read") or {}).items()})
             for r in config["rules"]]
    return rules, config.get("default", "break"), float(config.get("sample_sec", 2.0))


def classify_frame(frame: np.ndarray, rules: list[Rule], default: str, reader) -> dict:
    """화면 한 장 → {kind, (value 이름): 글자, (read 이름): 글자}."""
    for rule in rules:
        text = rule.spot.read(frame, reader)
        if rule.spot.matches(text):
            out = {"kind": rule.kind}
            if rule.value:
                out[rule.value] = text
            for name, spot in rule.read.items():
                t = spot.read(frame, reader)
                if spot.matches(t):
                    out[name] = t
            return out
    return {"kind": default}


def clock_sec(text: str | None) -> float | None:
    """'12:34' → 754.0."""
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", text or "")
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def label_segments(segments: list[dict], samples: list[dict], default: str) -> list[dict]:
    """segment 마다 안에 든 화면들의 다수결로 kind 를 붙인다. 안에 화면이 없으면(짧은 segment) 가장 가까운 화면.

    Args:
        segments: [{start_sec, end_sec, ...}] (장면 전환 사이 구간)
        samples: [{t, kind, game_time?}] 시간순.

    Returns:
        segments 에 kind, (live 면) game_start · game_end(경기 시간 글자)를 더한 새 목록.
    """
    out, k = [], 0
    for seg in segments:
        while k < len(samples) and samples[k]["t"] < seg["start_sec"]:
            k += 1
        inside, j = [], k
        while j < len(samples) and samples[j]["t"] < seg["end_sec"]:
            inside.append(samples[j])
            j += 1
        if not inside and samples:
            mid = (seg["start_sec"] + seg["end_sec"]) / 2
            near = [samples[x] for x in (k - 1, k) if 0 <= x < len(samples)]
            inside = [min(near, key=lambda s: abs(s["t"] - mid))]
        kind = Counter(s["kind"] for s in inside).most_common(1)[0][0] if inside else default
        new = {**seg, "kind": kind}
        clocks = [s["game_time"] for s in inside if s["kind"] == "live" and s.get("game_time")]
        if kind == "live" and clocks:
            new["game_start"], new["game_end"] = clocks[0], clocks[-1]
        out.append(new)
    return out


def game_range(samples: list[dict], step: float, header: dict | None = None, max_tail_sec: float = 30.0) -> dict | None:
    """경기 구간 (영상 초). live 화면이 없으면 None.

    - start_sec: 첫 live 화면 시각 - 그때 경기 시간 (경기 00:00 이 영상 몇 초인지)
    - end_sec: 마지막 live 화면 시각 + (header 경기 길이 - 그때 경기 시간). 차이가 max_tail_sec 를 넘거나
      header 가 없으면 마지막 live 화면 시각 + step
    - postgame 화면이 있으면 거기서 읽은 경기 길이·승자를 header 와 맞춰 본다 (checks)
    """
    live = [(s["t"], g) for s in samples if s["kind"] == "live" and (g := clock_sec(s.get("game_time"))) is not None]
    if not live:
        return None
    (t0, g0), (t1, g1) = live[0], live[-1]
    header = header or {}
    length = (header.get("result") or {}).get("length_sec")
    tail = length - g1 if length is not None else None
    end = t1 + tail if tail is not None and 0 <= tail <= max_tail_sec else t1 + step
    out = {"start_sec": round(max(0.0, t0 - g0), 3), "end_sec": round(end, 3),
           "last_clock": f"{int(g1 // 60):02d}:{int(g1 % 60):02d}", "header_length_sec": length}

    post = [s for s in samples if s["kind"] == "postgame"]
    if post:
        lengths = Counter(v for s in post if (v := clock_sec(s.get("game_length"))) is not None)
        results = Counter((_result(s.get("blue_result")), _result(s.get("red_result"))) for s in post)
        winner = next((("blue" if b == "WIN" else "red") for (b, r), _ in results.most_common()
                       if {b, r} == {"WIN", "LOSS"}), None)
        ocr_len = lengths.most_common(1)[0][0] if lengths else None
        out["postgame"] = {"start_sec": post[0]["t"], "game_length_sec": ocr_len, "winner": winner}
        h_winner = (header.get("result") or {}).get("winner")
        out["checks"] = {
            "length_matches_header": None if ocr_len is None or length is None else abs(ocr_len - length) <= 1,
            "winner_matches_header": None if winner is None or h_winner is None else winner == h_winner,
        }
    return out


def _result(text: str | None) -> str | None:
    t = _norm(text or "")
    if _contains(t, "WIN") and len(t) <= 4:
        return "WIN"
    if _contains(t, "LOSS") or t in ("LOS", "OSS"):
        return "LOSS"
    return None
