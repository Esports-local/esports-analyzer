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

사람 검수 (esports-api 가 보냄)
- text_overrides {대사 id: 고친 문장}: ASR 원문 대신 이 문장으로 번역한다 (앞뒤 문맥 줄에도 적용)
- 번역할 때 쓴 문장을 결과 sources 에 남기고, 문장이 바뀐 대사는 기존 번역을 다시 쓰지 않는다

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
    "text_overrides": {},        # {대사 id(문자열): 사람이 고친 문장} — ASR 원문 대신 번역
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
    plan = _plan(video, work_dir, p)
    client = Gemini(p["model"])
    if p["dry_run"]:
        counted = _count(client, plan)
        log.info("translate (dry_run): 대사 %d줄, 입력 토큰 %s, 합계 %d",
                 len(plan["targets"]), counted, sum(counted.values()))
        return None

    out, usage = plan["out"], TranslateUsage()
    for line_id, cond, body in _requests(plan):
        text, used = client.generate(body)
        out[str(line_id)][cond] = text
        usage.requests += 1
        usage.input_tokens += used.get("promptTokenCount", 0)
        usage.output_tokens += used.get("candidatesTokenCount", 0) + used.get("thoughtsTokenCount", 0)
        log.info("translate: %s %s %s", line_id, cond, text[:80])
    log.info("translate: 대사 %d줄, 요청 %d번, 입력 %d / 출력 %d 토큰",
             len(plan["targets"]), usage.requests, usage.input_tokens, usage.output_tokens)
    lines = {k: v for k, v in out.items() if v}
    result = TranslateResult(model=p["model"], conditions=plan["conds"], usage=usage, lines=lines,
                             sources={k: plan["texts"][k] for k in lines})
    return save_stage(work_dir, "translate", result, p)


def estimate(video: Path, work_dir: Path, params: dict | None = None, sample: int = 20) -> dict:
    """번역하지 않고 입력 토큰만 센다 (Gemini countTokens, 무료). 이미 번역한 (대사, 조건)은 빠진다.

    대사가 sample 줄보다 많으면 고르게 sample 줄만 세고 비율로 늘린다 (요청이 수백 번이면 몇 분 걸려서).

    Returns:
        {"model", "lines": 대상 대사 수, "requests": 보낼 요청 수, "input_tokens": {조건: 토큰},
         "sampled_lines": 실제로 센 대사 수}
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    plan = _plan(video, work_dir, p)
    todo = list(_todo(plan))
    requests = {c: sum(1 for _, t in todo if c in t) for c in plan["conds"]}
    picked = todo if len(todo) <= sample else [todo[round(i * (len(todo) - 1) / (sample - 1))] for i in range(sample)]
    counted = _count(Gemini(p["model"]), {**plan, "targets": [line for line, _ in picked]})
    seen = {c: sum(1 for _, t in picked if c in t) for c in plan["conds"]}
    tokens = {c: round(counted[c] * requests[c] / seen[c]) if seen[c] else 0 for c in plan["conds"]}
    return {"model": p["model"], "lines": len(plan["targets"]), "requests": sum(requests.values()),
            "input_tokens": tokens, "sampled_lines": len(picked)}


def _plan(video: Path, work_dir: Path, p: dict) -> dict:
    """번역할 대사·조건과 필요한 입력을 모은다.

    Raises:
        ValueError: 알 수 없는 조건.
        FileNotFoundError: 필요한 앞 단계 결과가 없음.
    """
    conds = [c for c in CONDITIONS if c in p["conditions"]]
    if not conds or set(p["conditions"]) - set(CONDITIONS):
        raise ValueError(f"conditions 는 {CONDITIONS} 중에서: {p['conditions']}")
    asr_lines = _read_result(work_dir, "asr")["lines"]
    lines = apply_overrides(asr_lines, p["text_overrides"] or {})
    runs = _read_result(work_dir, "roi_ocr")["runs"] if "B" in conds else []
    targets = select_lines(lines, p["line_ids"], p["max_lines"])
    prev, prev_sources = _previous(work_dir, p["model"]) if p["reuse"] else ({}, {})
    asr_texts = {str(x["id"]): x["text"] for x in asr_lines}
    texts = {str(x["id"]): x["text"] for x in lines}
    # 기존 번역은 그때 번역한 문장이 지금 문장과 같을 때만 다시 쓴다 (sources 가 없던 옛 결과는 ASR 원문으로 본다)
    out = {}
    for t in targets:
        k = str(t["id"])
        same = prev_sources.get(k, asr_texts.get(k)) == texts[k]
        out[k] = dict(prev.get(k, {})) if same else {}
    return {"p": p, "video": video, "conds": conds, "lines": lines, "runs": runs, "statics": static_values(runs),
            "targets": targets, "index": {line["id"]: i for i, line in enumerate(lines)}, "out": out, "texts": texts}


def apply_overrides(lines: list[dict], overrides: dict) -> list[dict]:
    """대사 문장을 사람이 고친 문장으로 바꾼 새 목록. overrides 키는 대사 id (문자열 또는 정수)."""
    fixed = {str(k): v for k, v in overrides.items() if isinstance(v, str) and v.strip()}
    return [{**x, "text": fixed[str(x["id"])].strip()} if str(x["id"]) in fixed else x for x in lines]


def _todo(plan: dict):
    """아직 번역 안 한 (대사, [조건…])."""
    for line in plan["targets"]:
        todo = [c for c in plan["conds"] if c not in plan["out"][str(line["id"])]]
        if todo:
            yield line, todo


def _requests(plan: dict):
    """(대사 id, 조건, 요청 본문). 조건 C 프레임은 대사마다 한 번만 뽑는다."""
    p = plan["p"]
    for line, todo in _todo(plan):
        frames = extract_frames(plan["video"], line, p["frames"], p["frame_size"]) if "C" in todo else []
        for cond in todo:
            yield line["id"], cond, build_request(plan["lines"], plan["index"][line["id"]], cond, p,
                                                  plan["runs"], plan["statics"], frames)


def _count(client, plan: dict) -> dict[str, int]:
    counted = {c: 0 for c in plan["conds"]}
    for _, cond, body in _requests(plan):
        counted[cond] += client.count_tokens(body)
    return counted


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


def _previous(work_dir: Path, model: str) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """같은 모델로 만든 기존 번역 ({대사 id: {조건: 번역}}, {대사 id: 번역한 문장}). 없거나 모델이 다르면 빈 dict."""
    path = stage_path(work_dir, "translate")
    if not path.exists():
        return {}, {}
    data = json.loads(path.read_text(encoding="utf-8"))
    result = data.get("result") or {}
    if result.get("model", data.get("params", {}).get("model")) != model:
        return {}, {}
    return result.get("lines", {}), result.get("sources", {})
