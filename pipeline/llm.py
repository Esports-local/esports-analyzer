"""Gemini REST 호출 (httpx). 키는 환경 변수 GEMINI_API_KEY (analyzer .env, compose 가 넘김).

JSON 으로만 답하게 할 때는 generate_json(system, user, schema): generationConfig 에 응답 스키마를 지정해
그 모양으로만 답하게 한다. 429/500/503 은 기다렸다 다시 보낸다.
"""

import json
import os
import time

import httpx

API = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-3.8-flash"


def available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


class Gemini:
    def __init__(self, model: str = DEFAULT_MODEL, retries: int = 4, timeout: float = 120):
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY 가 없습니다 (analyzer .env)")
        self.model = model
        self.retries = retries
        self.client = httpx.Client(headers={"x-goog-api-key": key}, timeout=timeout)

    def _post(self, method: str, body: dict) -> dict:
        r = None
        for attempt in range(self.retries):
            r = self.client.post(f"{API}/{self.model}:{method}", json=body)
            if r.status_code in (429, 500, 503):
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"Gemini {method} {r.status_code}: {r.text[:300]}")
            return r.json()
        raise RuntimeError(f"Gemini {method}: 다시 시도해도 실패 ({r.status_code if r else '-'})")

    def generate_json(self, system: str, user: str, schema: dict, temperature: float = 0.3) -> tuple[str, dict]:
        """(응답 글 원문, usageMetadata). 글은 schema 모양의 JSON 이어야 한다 (읽기·검사는 부르는 쪽)."""
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": temperature, "responseMimeType": "application/json",
                                     "responseSchema": schema}}
        d = self._post("generateContent", body)
        parts = (d.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
        text = "".join(x.get("text", "") for x in parts if not x.get("thought")).strip()
        return text, d.get("usageMetadata", {})


def parse_json(text: str) -> dict | None:
    """응답 글 → dict. 못 읽으면 None (잘린 응답 등)."""
    try:
        out = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return out if isinstance(out, dict) else None
