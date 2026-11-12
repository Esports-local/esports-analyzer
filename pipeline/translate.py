"""translate 단계: 해설 대사(asr)를 조건별로 영어 번역한다 (Gemini).

조건 (docs/PLAN.md 2장). 시스템 프롬프트·모델·설정은 모든 조건이 같고, 사용자 메시지에 붙는 정보만 다르다.
- A: 대사 + 앞뒤 문맥
- B: A + 그 시점 화면 OCR (팀, 선수 이름, 킬, KDA, 게임 시간)  ← work/roi_ocr.json 필요
- C: A + 그 시점 영상 프레임 (기본 2장)                          ← 원본 영상 필요

비용
- 기본 조건은 A, B. C(이미지)는 입력 토큰이 많아 params["conditions"] 로 직접 켠다
- dry_run: 요청을 만들고 Gemini countTokens(무료)로 입력 토큰만 세고, 번역 결과 파일은 만들지 않는다
- 이미 번역한 (대사, 조건)은 기존 work/translate.json 에서 다시 쓴다 (같은 모델일 때)
- line_ids / max_lines 로 번역할 대사를 줄일 수 있다

API 키: 환경변수 GEMINI_API_KEY (docker run --env-file .env)
"""

import base64
import io
import json
import logging
import os
import subprocess
import time
from pathlib import Path

import httpx
from PIL import Image

from pipeline.io import save_stage, stage_path
from pipeline.schema import TranslateResult, TranslateUsage
from pipeline.screen import snapshot, static_values

log = logging.getLogger(__name__)

API = "https://generativelanguage.googleapis.com/v1beta/models"
CONDITIONS = ("A", "B", "C")

DEFAULT_PARAMS = {
    "model": "gemini-3.8-flash",
    "conditions": ["A", "B"],
    "context_lines": 2,          # 앞뒤로 같이 보낼 대사 수
    "frames": 2,                 # 조건 C: 대사 시작 + 중간(최대 +1초)
    "frame_size": [960, 540],
    "temperature": 0.0,
    "thinking_level": "low",     # gemini-3.x. minimal 은 이 모델에서 안 됨
    "max_output_tokens": 1000,   # 생각 토큰 포함
    "line_ids": None,            # 이 대사만 번역 (None 이면 전부)
    "max_lines": None,           # 앞에서부터 이만큼만
    "reuse": True,               # 기존 translate.json 에 있는 번역은 다시 요청하지 않음
    "dry_run": False,            # 입력 토큰만 세고 끝
}

SYSTEM_PROMPT = (
    "You translate Korean League of Legends esports commentary into natural English.\n"
    "- The Korean text is speech-recognition output, so player, team, champion and skill names may be misheard.\n"
    "- When broadcast screen information is given, use the names exactly as they appear there.\n"
    "- Translate only the TARGET line; the context lines are for reference.\n"
    "- Output only the English translation of the TARGET line."
)
SCREEN_INTRO = ("Broadcast screen text at this moment (OCR of the HUD: teams, player names by side/role, "
                "kills, KDA):")
FRAMES_INTRO = "Broadcast frames at this moment:"


def run(video: Path, work_dir: Path, params: dict | None = None) -> Path | None:
    """translate 단계를 실행한다.

    Args:
        video: 원본 영상 경로 (조건 C 프레임용).
        work_dir: 작업 디렉터리. work/asr.json 이 있어야 하고, 조건 B 는 work/roi_ocr.json 도 필요하다.
        params: DEFAULT_PARAMS 를 덮어쓸 설정.

    Returns:
        work/translate.json 경로. dry_run 이면 저장하지 않고 None.

    Raises:
        ValueError: 알 수 없는 조건.
        FileNotFoundError: 필요한 앞 단계 결과가 없음.
        RuntimeError: GEMINI_API_KEY 가 없거나 API 오류.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    conds = [c for c in CONDITIONS if c in p["conditions"]]
    if not conds or set(p["conditions"]) - set(CONDITIONS):
        raise ValueError(f"conditions 는 {CONDITIONS} 중에서: {p['conditions']}")
    lines = _read_result(work_dir, "asr")["lines"]
    runs = _read_result(work_dir, "roi_ocr")["runs"] if "B" in conds else []
    statics = static_values(runs)
    targets = select_lines(lines, p["line_ids"], p["max_lines"])

    prev = _previous(work_dir, p["model"]) if p["reuse"] else {}
    client = Gemini(p["model"])
    out: dict[str, dict[str, str]] = {str(line["id"]): dict(prev.get(str(line["id"]), {})) for line in targets}
    usage = TranslateUsage()
    counted = {c: 0 for c in conds}
    by_id = {line["id"]: i for i, line in enumerate(lines)}

    for line in targets:
        done = out[str(line["id"])]
        todo = [c for c in conds if c not in done]
        if not todo:
            continue
        frames = extract_frames(video, line, p["frames"], p["frame_size"]) if "C" in todo else []
        for cond in todo:
            body = build_request(lines, by_id[line["id"]], cond, p, runs, statics, frames)
            if p["dry_run"]:
                counted[cond] += client.count_tokens(body)
                continue
            text, used = client.generate(body)
            done[cond] = text
            usage.requests += 1
            usage.input_tokens += used.get("promptTokenCount", 0)
            usage.output_tokens += used.get("candidatesTokenCount", 0) + used.get("thoughtsTokenCount", 0)
            log.info("translate: %s %s %s", line["id"], cond, text[:80])

    if p["dry_run"]:
        log.info("translate (dry_run): 대사 %d줄, 입력 토큰 %s, 합계 %d", len(targets), counted, sum(counted.values()))
        return None
    log.info("translate: 대사 %d줄, 요청 %d번, 입력 %d / 출력 %d 토큰",
             len(targets), usage.requests, usage.input_tokens, usage.output_tokens)
    result = TranslateResult(model=p["model"], conditions=conds, lines={k: v for k, v in out.items() if v},
                             usage=usage)
    return save_stage(work_dir, "translate", result, p)


def select_lines(lines: list[dict], line_ids, max_lines) -> list[dict]:
    """번역할 대사를 고른다."""
    picked = [line for line in lines if not line_ids or line["id"] in set(line_ids)]
    return picked[:max_lines] if max_lines else picked


def build_request(lines: list[dict], i: int, cond: str, p: dict, runs: list[dict], statics: dict,
                  frames: list[str]) -> dict:
    """대사 하나 × 조건 하나의 generateContent 요청 본문."""
    n = p["context_lines"]
    line = lines[i]
    before = "\n".join(x["text"] for x in lines[max(0, i - n):i])
    after = "\n".join(x["text"] for x in lines[i + 1:i + 1 + n])
    parts = [{"text": f"Context before:\n{before}\n\nTARGET:\n{line['text']}\n\nContext after:\n{after}"}]
    if cond == "B":
        screen = snapshot(runs, (line["start_sec"] + line["end_sec"]) / 2, statics)
        parts.append({"text": SCREEN_INTRO + "\n" + json.dumps(screen, ensure_ascii=False)})
    elif cond == "C":
        parts.append({"text": FRAMES_INTRO})
        parts += [{"inline_data": {"mime_type": "image/jpeg", "data": f}} for f in frames]
    return {"systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": p["temperature"], "maxOutputTokens": p["max_output_tokens"],
                                 "thinkingConfig": {"thinkingLevel": p["thinking_level"]}}}


def extract_frames(video: Path, line: dict, count: int, size: list[int]) -> list[str]:
    """대사 시작부터 중간(최대 +1초)까지 count 장을 JPEG base64 로."""
    start, end = line["start_sec"], line["end_sec"]
    last = min(end, (start + end) / 2 + 1)
    times = [start + (last - start) * k / max(1, count - 1) for k in range(count)]
    out = []
    for t in times:
        png = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1",
                              "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True, check=True).stdout
        im = Image.open(io.BytesIO(png)).convert("RGB").resize(tuple(size), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
        out.append(base64.b64encode(buf.getvalue()).decode())
    return out


class Gemini:
    """Gemini REST 호출. 429/503 은 기다렸다 다시 보낸다."""

    def __init__(self, model: str, retries: int = 5):
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY 가 없습니다 (docker run --env-file .env)")
        self.model = model
        self.retries = retries
        self.client = httpx.Client(headers={"x-goog-api-key": key}, timeout=120)

    def _post(self, method: str, body: dict) -> dict:
        for attempt in range(self.retries):
            r = self.client.post(f"{API}/{self.model}:{method}", json=body)
            if r.status_code in (429, 500, 503):
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"Gemini {method} {r.status_code}: {r.text[:300]}")
            return r.json()
        raise RuntimeError(f"Gemini {method}: 다시 시도해도 실패 ({r.status_code})")

    def count_tokens(self, body: dict) -> int:
        """입력 토큰 수 (무료)."""
        req = {"model": f"models/{self.model}", **{k: v for k, v in body.items() if k != "generationConfig"}}
        return self._post("countTokens", {"generateContentRequest": req})["totalTokens"]

    def generate(self, body: dict) -> tuple[str, dict]:
        """(번역문, usageMetadata)."""
        d = self._post("generateContent", body)
        parts = (d.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
        text = "".join(x.get("text", "") for x in parts if not x.get("thought")).strip()
        return text, d.get("usageMetadata", {})


def _read_result(work_dir: Path, stage: str) -> dict:
    path = stage_path(work_dir, stage)
    if not path.exists():
        raise FileNotFoundError(f"{stage} 결과가 없습니다: {path} ({stage} 단계를 먼저 실행)")
    return json.loads(path.read_text(encoding="utf-8"))["result"]


def _previous(work_dir: Path, model: str) -> dict[str, dict[str, str]]:
    """같은 모델로 만든 기존 번역 {대사 id: {조건: 번역}}. 없거나 모델이 다르면 빈 dict."""
    path = stage_path(work_dir, "translate")
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if (data.get("result") or {}).get("model", data.get("params", {}).get("model")) != model:
        return {}
    return (data.get("result") or {}).get("lines", {})
