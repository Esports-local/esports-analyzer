"""장면형 숏폼: LLM 이 후보마다 자를 문장 범위 · 제목 · 순위를 고르고, 코드가 검사해 최종값을 정한다.

입력은 llm_input.build (analyzer.json 모양, 후보 장면만). LLM 은 경기당 한 번, 응답 스키마를 지정해 JSON 으로만 받는다.
같은 입력·모델·프롬프트 버전이면 지난 응답을 다시 쓴다 (work/shorts_llm.json 의 input_hash).

검사 (checks: result = pass · warn · fixed)
- 고치는 것: lines_in_scene(문장 번호가 그 장면 안, 시작 ≤ 끝 → 아니면 장면 전체) · covers_events(킬·오브젝트가
  2초 여유와 함께 들어가게 넓힘) · length(8초 미만이면 장면 전체) · title_rules(10자 이내, 이모지·#·"공식" 없음,
  겹치지 않음 → 걸린 제목은 버리고 다 버려지면 자동 제목)
- 표시만: length(180초 넘음, 쇼츠로 못 올림) · start_not_in_fight · end_not_in_fight(바로 앞·뒤 문장이 외침) ·
  title_names(제목의 선수·챔피언이 그 장면 이벤트에 없음) · title_counts(제목의 킬 수가 다름) · rank_vs_rule(LLM 1위가 규칙 점수 하위권)
LLM 이 없거나 실패하면 규칙 순서 · 장면 전체 · 자동 제목 (final.source = rule).
"""

import hashlib
import json
import logging
import re
from datetime import datetime, timezone

from pipeline import llm

log = logging.getLogger(__name__)

PROMPT_VERSION = "v1"
TITLE_MAX = 10
MIN_SEC, SHORTS_MAX_SEC = 8.0, 180.0
EVENT_LEAD_SEC = 2.0     # 킬·오브젝트 앞에 남길 여유
LINE_PAD_SEC = 0.5       # 문장 앞뒤로 더 남기는 초
SHOUT = 0.9              # 이 소리 크기 이상이면 외침 (전투 중)

SYSTEM_PROMPT = f"""너는 LoL 중계 숏폼 편집자다. 입력은 한 세트 경기의 숏폼 후보 장면 목록(JSON)이다.
입력 모양은 header.segment_types 에 설명돼 있다 (scene → cut → events · lines).
후보 장면(scene)마다 (1) 어디서 자를지, (2) 제목, (3) 순위를 정해 지정한 JSON 으로만 답한다.

[입력에서 알아 둘 것]
- events 의 킬·오브젝트 t 는 실제로 일어난 시각이다. 선수·챔피언 이름은 header.players 와 events 에 있는 것만 쓴다.
- lines 는 해설 받아쓰기라 오타와 잘못 들은 이름이 많다. loud 는 그 문장 소리 크기(0~1)이고 외침일수록 크다.
- gold_diff 는 블루 기준 골드 차이다. 1000 미만 변화는 무시하고, 부호가 바뀌었을 때만 역전이다.

[자르기] 전투가 중심이다.
- start_line: 전투가 시작되기 직전, 상황을 알려 주는 문장 1개부터. 그 앞의 운영·밴픽·잡담 설명은 뺀다.
- end_line: 마지막 킬·오브젝트 뒤 해설 반응이나 정리 문장까지. 다음 이야기(다른 라인, 다음 계획)로 넘어가는 문장은 뺀다.
- 그 장면의 모든 킬·오브젝트가 구간 안에 들어가야 한다. 문장 번호(lines.id)는 그 장면 안에서만 고른다.
- 소리가 커지는 문장(싸움 시작)과 작아지는 문장(정리)을 참고한다.

[제목]
- 한국어, 공백 포함 {TITLE_MAX}자 이내, 3개. 첫 번째가 가장 좋은 것.
- 선수 이름 또는 챔피언 하나 + 무슨 일이 있었는지 한 마디. 조사는 줄인다. 예: "듀로의 즉시 복수", "15분 바텀 교환", "카나비 칼각"
- 실제로 일어난 일(events)만 쓴다. 없는 킬·역전·펜타킬을 지어내지 않는다. 팀 이름 "A vs B" 는 넣지 않는다.
- 이모지 · 해시태그 · "공식" 같은 표현은 쓰지 않는다.

[순위]
- 킬 수와 짧은 시간 안 연속 킬, 퍼스트 블러드·용·바론 같은 큰 오브젝트, 해설 반응(외침이 몰린 정도, loud),
  골드 차이가 크게 바뀌거나 뒤집힌 것을 함께 본다. 1이 가장 좋다. 순위는 겹치지 않는다.
- reason 에 한 줄로 이유를 쓴다.

모든 후보에 답한다."""

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {"candidates": {"type": "ARRAY", "items": {
        "type": "OBJECT",
        "properties": {"id": {"type": "STRING"}, "rank": {"type": "INTEGER"},
                       "start_line": {"type": "INTEGER"}, "end_line": {"type": "INTEGER"},
                       "titles": {"type": "ARRAY", "items": {"type": "STRING"}},
                       "reason": {"type": "STRING"}},
        "required": ["id", "rank", "start_line", "end_line", "titles", "reason"]}}},
    "required": ["candidates"],
}

OBJECT_SCORE = {"baron": 3, "elder": 3, "dragon": 2, "atakhan": 2, "herald": 1, "grubs": 0}
BANNED = ("공식", "#")
EMOJI_RE = re.compile("[\U0001F000-\U0001FFFF☀-➿️]")
COUNT_WORDS = {"더블킬": 2, "트리플킬": 3, "쿼드라킬": 4, "펜타킬": 5}


def run(llm_input: dict, previous: dict | None = None, client=None, model: str = llm.DEFAULT_MODEL,
        allow_call: bool = True) -> dict:
    """LLM 을 부르고(같은 입력이면 지난 응답 재사용) 검사한 결과 (work/shorts_llm.json 내용).
    allow_call=False 면 부르지 않는다 (지난 응답이 같으면 그것은 쓴다)."""
    text = json.dumps(llm_input, ensure_ascii=False, separators=(",", ":"))
    input_hash = hashlib.sha256(f"{PROMPT_VERSION}|{model}|{text}".encode()).hexdigest()[:16]
    out = {"model": model, "prompt_version": PROMPT_VERSION, "input_hash": input_hash,
           "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "raw_response": None, "usage": {}, "reused": False, "error": None}
    if previous and previous.get("input_hash") == input_hash and previous.get("raw_response"):
        out.update(raw_response=previous["raw_response"], usage=previous.get("usage", {}), reused=True,
                   created_at=previous.get("created_at", out["created_at"]))
    elif not allow_call:
        out["error"] = "LLM 을 부르지 않음 (use_llm 꺼짐)"
    elif client is not None or llm.available():
        try:
            out["raw_response"], out["usage"] = (client or llm.Gemini(model)).generate_json(
                SYSTEM_PROMPT, text, RESPONSE_SCHEMA)
        except Exception as e:  # 연결 실패, 키 오류 → 규칙 결과
            out["error"] = f"LLM 호출 실패: {e}"
    else:
        out["error"] = "GEMINI_API_KEY 가 없어 LLM 을 부르지 않음"
    parsed = llm.parse_json(out["raw_response"]) if out["raw_response"] else None
    if out["raw_response"] and parsed is None:
        out["error"] = "응답을 읽지 못함"
    out["candidates"] = decide(llm_input, (parsed or {}).get("candidates") or [])
    return out


# ---------------------------------------------------------------- 검사 · 최종값

def decide(llm_input: dict, answers: list[dict]) -> list[dict]:
    """후보마다 {id, llm, checks, final}. final.rank 는 1부터 겹치지 않게."""
    scenes = llm_input.get("segments") or []
    names = _player_names(llm_input)
    rule_order = sorted(scenes, key=lambda s: -rule_score(s))
    rule_rank = {s["id"]: i + 1 for i, s in enumerate(rule_order)}
    by_id: dict[str, dict] = {}
    for a in answers:
        if isinstance(a, dict) and a.get("id") in rule_rank and a["id"] not in by_id:
            by_id[a["id"]] = a
    ranks = [by_id[s["id"]].get("rank") for s in scenes if s["id"] in by_id]
    llm_ranks_ok = len(by_id) == len(scenes) and sorted(ranks) == list(range(1, len(scenes) + 1))

    out, used_titles = [], set()
    order = sorted(scenes, key=lambda s: (by_id[s["id"]]["rank"] if llm_ranks_ok else rule_rank[s["id"]]))
    for i, s in enumerate(order, 1):
        a = by_id.get(s["id"])
        checks: list[dict] = []
        start, end, line_range, fixed = _span(s, a, checks)
        titles, title_fixed = _titles(s, a, used_titles, checks)
        title = titles[0] if titles else auto_title(s, names)
        used_titles.add(title)
        _warns(s, a, start, end, title, names, checks, i if llm_ranks_ok else None, rule_rank[s["id"]], len(scenes))
        if a is None:
            source = "rule"
        else:
            source = "llm+fixed" if fixed or title_fixed or not llm_ranks_ok else "llm"
        out.append({"id": s["id"], "llm": a, "checks": checks,
                    "final": {"rank": i, "start_sec": round(start, 3), "end_sec": round(end, 3),
                              "line_range": line_range, "title": title, "titles": titles or [title],
                              "reason": (a or {}).get("reason") or "", "source": source,
                              "warnings": [c["detail"] for c in checks if c["result"] == "warn"]}})
    return out


def rule_score(scene: dict) -> float:
    """LLM 이 없을 때 순위: 킬 · 태그 · 오브젝트 · 외침 · 골드 변화."""
    evs = _events(scene)
    score = 0.0
    for e in evs:
        if e["type"] == "kill":
            score += 2 + (1 if e.get("tags") else 0)
        elif e["type"] == "objective":
            score += OBJECT_SCORE.get((e.get("subtype") or "").split(":")[0], 1)
    louds = [l.get("loud") or 0 for l in _lines(scene)]
    score += 2 * max(louds, default=0)
    g = scene["fields"].get("gold_diff") or {}
    if g.get("start") is not None and g.get("end") is not None:
        score += min(abs(g["end"] - g["start"]) / 1000, 2)
    return round(score, 3)


def _span(s: dict, a: dict | None, checks: list) -> tuple[float, float, list | None, bool]:
    f = s["fields"]
    a0, b0 = f["from_sec"], f["to_sec"]
    lines = _lines(s)
    by_id = {l["id"]: l for l in lines}
    if a is None:
        return a0, b0, None, False
    sl, el = by_id.get(a.get("start_line")), by_id.get(a.get("end_line"))
    if not sl or not el or sl["from_sec"] > el["from_sec"]:
        checks.append({"name": "lines_in_scene", "result": "fixed",
                       "detail": f"문장 {a.get('start_line')}~{a.get('end_line')} 이 장면에 없거나 순서가 틀려 장면 전체로"})
        return a0, b0, None, True
    checks.append({"name": "lines_in_scene", "result": "pass"})
    start, end = max(a0, sl["from_sec"] - LINE_PAD_SEC), min(b0, el["to_sec"] + LINE_PAD_SEC)
    fixed = False
    missed = []
    for e in _events(s):
        if e["type"] not in ("kill", "objective"):
            continue
        t = e["t"]
        if start > t - EVENT_LEAD_SEC:
            start = max(a0, _line_start_before(lines, t - EVENT_LEAD_SEC))
            missed.append(e.get("game_time") or f"{t:.0f}")
        if end < t + 1:
            end = min(b0, _line_end_after(lines, t + 1))
            missed.append(e.get("game_time") or f"{t:.0f}")
    if missed:
        fixed = True
        checks.append({"name": "covers_events", "result": "fixed",
                       "detail": f"{', '.join(sorted(set(missed)))} 이벤트가 잘려 넓힘"})
    else:
        checks.append({"name": "covers_events", "result": "pass"})
    if end - start < MIN_SEC:
        checks.append({"name": "length", "result": "fixed", "detail": f"{end - start:.1f}초라 장면 전체로"})
        return a0, b0, None, True
    if end - start > SHORTS_MAX_SEC:
        checks.append({"name": "length", "result": "warn", "detail": f"{end - start:.0f}초, 쇼츠(3분)로는 못 올림"})
    else:
        checks.append({"name": "length", "result": "pass"})
    return start, end, [a["start_line"], a["end_line"]], fixed


def _titles(s: dict, a: dict | None, used: set, checks: list) -> tuple[list[str], bool]:
    if a is None:
        return [], False
    keep, dropped = [], []
    for t in a.get("titles") or []:
        t = (t or "").strip()
        why = None
        if not t:
            why = "빈 제목"
        elif len(t) > TITLE_MAX:
            why = f"{len(t)}자"
        elif EMOJI_RE.search(t) or any(b in t for b in BANNED):
            why = "이모지·해시태그·공식 표현"
        elif t in keep or t in used:
            why = "겹침"
        if why:
            dropped.append(f"'{t}'({why})")
        else:
            keep.append(t)
    if dropped:
        checks.append({"name": "title_rules", "result": "fixed",
                       "detail": "버린 제목: " + ", ".join(dropped) + ("" if keep else " → 자동 제목")})
    else:
        checks.append({"name": "title_rules", "result": "pass"})
    return keep, bool(dropped)


def _warns(s, a, start, end, title, names, checks, llm_rank, rule_rank, n):
    lines = _lines(s)
    inside = [l for l in lines if l["from_sec"] >= start - 0.01 and l["to_sec"] <= end + 0.01]
    if a is not None and inside:
        before = [l for l in lines if l["to_sec"] <= inside[0]["from_sec"] + 0.01 and l is not inside[0]]
        after = [l for l in lines if l["from_sec"] >= inside[-1]["to_sec"] - 0.01 and l is not inside[-1]]
        if before and (before[-1].get("loud") or 0) >= SHOUT:
            checks.append({"name": "start_not_in_fight", "result": "warn",
                           "detail": f"바로 앞 문장 {before[-1]['id']}가 외침(loud {before[-1]['loud']}), 전투 도중에 시작"})
        if after and (after[0].get("loud") or 0) >= SHOUT:
            checks.append({"name": "end_not_in_fight", "result": "warn",
                           "detail": f"바로 뒤 문장 {after[0]['id']}가 외침(loud {after[0]['loud']}), 전투가 끝나기 전에 끊김"})
    evs = [e for e in _events(s) if start <= e["t"] <= end]
    in_scene = set()
    for e in evs:
        for w in [e.get("killer")] + (e.get("victims") or []) + (e.get("assists") or []):
            if w:
                in_scene |= {x for x in names if x in w}
    stray = sorted(x for x in names if x in title and x not in in_scene)
    if stray:
        checks.append({"name": "title_names", "result": "warn", "detail": f"제목의 {', '.join(stray)} 는 이 장면 이벤트에 없음"})
    kills = sum(1 for e in evs if e["type"] == "kill")
    said = [int(m) for m in re.findall(r"(\d+)\s*킬", title)] + [v for k, v in COUNT_WORDS.items() if k in title]
    if said and max(said) > kills:
        checks.append({"name": "title_counts", "result": "warn", "detail": f"제목은 {max(said)}킬, 구간 안 킬은 {kills}개"})
    if llm_rank == 1 and n >= 3 and rule_rank > (n + 1) // 2:
        checks.append({"name": "rank_vs_rule", "result": "warn", "detail": f"LLM 1위, 규칙 점수로는 {rule_rank}위"})


def auto_title(s: dict, names: set | None = None) -> str:
    """자동 제목 (10자 안쪽)."""
    evs = _events(s)
    kills = [e for e in evs if e["type"] == "kill"]
    objs = [(e.get("subtype") or "").split(":")[0] for e in evs if e["type"] == "objective"]
    label = {"dragon": "용 싸움", "baron": "바론 싸움", "elder": "장로 싸움", "herald": "전령", "grubs": "유충"}
    if len(kills) == 1 and not objs and kills[0].get("killer"):
        name = kills[0]["killer"].split(" ")[1] if kills[0]["killer"].count(" ") >= 1 else kills[0]["killer"]
        t = f"{name} 킬"
        return t if len(t) <= TITLE_MAX else "킬 장면"
    if objs:
        return label.get(objs[0], "오브젝트 싸움")
    if len({e.get("team") for e in kills}) > 1:
        return f"킬 교환 {len(kills)}"
    return f"킬 {len(kills)}" if kills else "장면"


def _lines(s: dict) -> list[dict]:
    return sorted((l for c in s.get("children", []) for l in c["fields"].get("lines", [])), key=lambda l: l["from_sec"])


def _events(s: dict) -> list[dict]:
    return sorted((e for c in s.get("children", []) for e in c["fields"].get("events", [])), key=lambda e: e["t"])


def _player_names(llm_input: dict) -> set[str]:
    """header.players "팀 이름 (챔피언)" → {이름, 챔피언}."""
    out = set()
    for v in ((llm_input.get("header") or {}).get("players") or {}).values():
        m = re.match(r"^(?:\S+ )?(\S+) \((.+)\)$", v or "")
        if m:
            out |= {m.group(1), m.group(2)}
    return out


def _line_start_before(lines: list[dict], t: float) -> float:
    starts = [l["from_sec"] for l in lines if l["from_sec"] <= t]
    return max(starts) if starts else t


def _line_end_after(lines: list[dict], t: float) -> float:
    ends = [l["to_sec"] for l in lines if l["to_sec"] >= t]
    return min(ends) if ends else t
