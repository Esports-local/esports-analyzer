"""analyzer.json: 단계 결과를 시간순 트리 하나로 묶는다 (sevenmin_master 의 mastering_data.json 방식).

run_pipeline 이 끝날 때마다 <outputs>/<영상>/analyzer.json 을 다시 만든다. 단계 파일(work/*.json)은 그대로 둔다.

    {schema_version,
     header:   {video_id, video_type, range, context(header 단계 결과), game, players, clock, segment_types},
     segments: [phase{fields, children: [cut{fields}]}]  시간순. 그 시간의 데이터는 전부 cut 안에 있다
    (분석 범위 밖 op.gg 기록은 넣지 않는다, 전체는 work/opgg_timeline.json)}

cut.fields: from_sec · to_sec · kind · transition_in · game_time
    events     우리 분석 이벤트 (선수 이름·챔피언 붙임). 리플레이 중이라 화면으로 못 잡은 op.gg 기록은
               source ["opgg"], filled: true 로 채운다 (검증 정확도에서 뺀다)
    lines      해설 대사 (교정 원문 raw · 내역 fixes)
    reference  op.gg 정답지: 킬·오브젝트(선수·챔피언·맵 구역·짝이 되는 우리 이벤트와 시각 차이), 1분 위치·골드
    signals    장면 길이만큼의 초 단위 소리 크기·골드 차이·컷 빈도 + 요약
    screen     장면 끝 화면 값 (킬 수, 골드 차이)
화면 OCR 원본은 넣지 않는다 (work/roi_ocr.json).

phase: 경기 구간(context.game) 안의 컷은 모두 game(리플레이·캠 포함), 밖은 화면 종류가 같은 컷끼리.
시각이 장면 사이(전환 구간)면 바로 앞 장면, 첫 장면 앞이면 첫 장면에 넣는다. 대사는 시작 시각 기준.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from pipeline.game.context import game_to_video

SEGMENT_TYPES = {
    "phase": {
        "description": "큰 구간. game(경기 00:00 ~ 끝, 리플레이·캠 포함) · draft(밴픽) · postgame(경기 뒤 통계·하이라이트) · "
                       "countdown(세트 사이 대기) · break(그 밖)",
        "fields": {"kind": {"type": "string"}, "from_sec": {"type": "float"}, "to_sec": {"type": "float"}},
        "allowed_children": {"cut": None},
    },
    "cut": {
        "description": "장면 전환(cut · dissolve · wipe) 사이의 연속 장면. 숏폼을 자를 때 경계로 쓴다",
        "fields": {
            "from_sec": {"type": "float"}, "to_sec": {"type": "float"},
            "kind": {"type": "string", "description": "live · replay · postgame · countdown · draft · break"},
            "transition_in": {"type": "string", "description": "이 장면으로 들어온 전환 cut · dissolve · wipe"},
            "game_time": {"type": "list[string]", "description": "live 면 장면 안에서 읽은 경기 시간 [처음, 끝]"},
            "events": {"type": "list[object]", "description": "킬·오브젝트·타워 (context.events), 앞뒤 해설 대사(lines) 포함"},
            "lines": {"type": "list[object]", "description": "해설 대사 {id, from_sec, to_sec, text, raw?, fixes?}"},
            "screen": {"type": "object", "description": "장면 끝 화면 값 {kills: {blue, red}, gold_diff}"},
            "reference": {"type": "list[object]", "description": "op.gg 정답지 (검증용): kill · 오브젝트(grubs·dragon·herald·baron·…) "
                          "{video_sec, game_time, side, killer, killer_name, killer_champion, position, area, matched_event, diff_sec}, "
                          "minute {positions, gold}"},
            "signals": {"type": "object", "description": "장면 길이만큼 초 단위 loudness(소리 크기 0~1)·gold_diff·cut_rate + loudness_max·loudness_mean"},

        },
        "allowed_children": {},
    },
}


def _load(work: Path, stage: str) -> dict | None:
    path = work / f"{stage}.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8")).get("result")
    return None if (result or {}).get("error") else result


MATCH_SEC = 10.0  # op.gg 기록과 우리 이벤트를 짝지을 최대 시각 차이


def build(work_dir: Path) -> dict:
    work = work_dir / "work"
    header, pre, ctx = _load(work, "header"), _load(work, "preprocess") or {}, _load(work, "context") or {}
    asr, roi, ref = _load(work, "asr") or {}, _load(work, "roi_ocr"), _load(work, "opgg_timeline")
    start = pre.get("range_start_sec") or 0.0
    end = pre.get("duration_sec") or max((s["end_sec"] for s in ctx.get("segments", [])), default=start)
    players, clock, signals = ctx.get("players", {}), ctx.get("clock", []), ctx.get("signals", {})

    cuts = [dict(s) for s in ctx.get("segments", [])] or [
        {"id": 1, "start_sec": start, "end_sec": end, "kind": None, "transition_in": None, "game_time": None}]
    nodes = [_cut(c) for c in cuts]
    lines = {ln["id"]: _line(ln) for ln in asr.get("lines", [])}
    events = [_event(e, players, lines) for e in ctx.get("events", [])]
    _attach(nodes, events, "events", lambda e: e["t"])
    _attach(nodes, list(lines.values()), "lines", lambda ln: ln["from_sec"])
    refs, _outside = _reference_items(ref, clock, players, events, start, end)
    _attach(nodes, refs, "reference", lambda r: r["video_sec"])
    _attach(nodes, _fill_from_reference(refs, nodes, lines), "events", lambda e: e["t"])
    for node in nodes:
        node["fields"]["events"].sort(key=lambda e: e["t"])
    for node in nodes:
        f = node["fields"]
        f["signals"] = _signals(signals, f["from_sec"], f["to_sec"])
        f["screen"] = _screen(roi, signals, f["to_sec"])

    return {
        "schema_version": 1,
        "header": {"video_id": work_dir.name, "video_type": "lol", "range": {"from_sec": start, "to_sec": end},
                   "context": header or {}, "game": ctx.get("game", {}), "players": players, "clock": clock,
                   "segment_types": SEGMENT_TYPES},
        "segments": _phases(nodes, ctx.get("game") or {}),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def write(work_dir: Path) -> Path:
    out = work_dir / "analyzer.json"
    out.write_text(json.dumps(build(work_dir), ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def _cut(c: dict) -> dict:
    return {"id": f"cut-{c['id']}", "type": "cut",
            "fields": {"from_sec": c["start_sec"], "to_sec": c["end_sec"], "kind": c.get("kind"),
                       "transition_in": c.get("transition_in"), "game_time": c.get("game_time"),
                       "events": [], "lines": [], "reference": []},
            "children": []}


def _line(ln: dict) -> dict:
    out = {"id": ln["id"], "from_sec": ln["start_sec"], "to_sec": ln["end_sec"], "text": ln["text"]}
    if ln.get("raw"):
        out["raw"], out["fixes"] = ln["raw"], ln.get("fixes", [])
    return out


def _attach(nodes: list[dict], items: list[dict], key: str, time_of) -> None:
    """items 를 시각이 속한 컷에 넣는다. 컷 사이(전환 구간)면 바로 앞 컷, 첫 컷 앞이면 첫 컷."""
    for item in sorted(items, key=time_of):
        t = time_of(item)
        target = nodes[0]
        for node in nodes:
            if node["fields"]["from_sec"] <= t:
                target = node
            else:
                break
        target["fields"][key].append(item)


def _screen(roi: dict | None, signals: dict, t: float) -> dict:
    """t 초(컷 끝) 즈음 화면 값: 팀 킬 수(마지막으로 읽힌 값), 골드 차이(signals)."""
    kills = {}
    for r in (roi or {}).get("runs", []):
        if r["tag"] == "team_kills" and r["start_sec"] < t and r["text"].isdigit():
            kills[r["slot"]] = int(r["text"])  # runs 는 시간순 → 마지막 값이 남는다
    gold = signals.get("gold_diff") or []
    i = int((t - signals.get("start_sec", 0.0)) // (signals.get("step_sec") or 1.0)) - 1
    return {"kills": kills or None, "gold_diff": gold[i] if 0 <= i < len(gold) else None}


def _phase_kind(cut: dict, game: dict) -> str:
    f = cut["fields"]
    if game.get("start_sec") is not None and game["start_sec"] <= f["from_sec"] < (game.get("end_sec") or float("inf")):
        return "game"
    return {"live": "game", "replay": "postgame"}.get(f["kind"], f["kind"] or "break")


def _phases(cuts: list[dict], game: dict) -> list[dict]:
    phases: list[dict] = []
    for c in cuts:
        kind = _phase_kind(c, game)
        if phases and phases[-1]["fields"]["kind"] == kind:
            phases[-1]["children"].append(c)
            phases[-1]["fields"]["to_sec"] = c["fields"]["to_sec"]
        else:
            phases.append({"id": f"phase-{len(phases) + 1}", "type": "phase",
                           "fields": {"kind": kind, "from_sec": c["fields"]["from_sec"], "to_sec": c["fields"]["to_sec"]},
                           "children": [c]})
    return phases


REF_SLACK_SEC = 20.0  # clock 구간 밖이어도 이 초 안이면 그 구간 차이로 옮긴다 (리플레이·캠 동안에도 경기는 흐른다)


def to_video(clock: list[dict], g: float, slack: float = REF_SLACK_SEC) -> float | None:
    """경기 g 초 → 영상 초. clock 구간 안이면 그대로, 아니면 경기 시각이 가장 가까운 구간(slack 안)의 차이로."""
    v = game_to_video(clock, g)
    if v is not None:
        return v
    best = None
    for c in clock:
        g0, g1 = c["game_start_sec"], c["game_start_sec"] + c["end_sec"] - c["start_sec"]
        d = g0 - g if g < g0 else g - g1 if g > g1 else 0.0
        if d <= slack and (best is None or d < best[0]):
            best = (d, g - c["game_start_sec"] + c["start_sec"])
    return best[1] if best else None


def _who(players: dict, slot: str | None) -> dict:
    p = players.get(slot or "", {})
    return {"name": p.get("name"), "champion": p.get("champion")}


def _event(e: dict, players: dict, lines: dict[int, dict]) -> dict:
    """context 이벤트에 선수 이름·챔피언과 앞뒤 대사(번호 대신 문장)를 붙인다."""
    out = {k: v for k, v in e.items() if k not in ("start_sec", "end_sec", "segment", "line_ids")}
    out["lines"] = [{k: v for k, v in lines[i].items() if k in ("id", "from_sec", "text")}
                    for i in e.get("line_ids", []) if i in lines]
    if e.get("killer"):
        out["killer_name"], out["killer_champion"] = _who(players, e["killer"]).values()
    out["victims_detail"] = [{"slot": s, **_who(players, s)} for s in e.get("victims", [])]
    return out


MAP_SIZE = 14870.0
DRAGON_PIT, BARON_PIT = (9866, 4414), (5007, 10471)


def map_area(pos: dict | None) -> str | None:
    """op.gg 맵 좌표 → 구역 (대략). 블루 진영이 왼쪽 아래."""
    if not pos:
        return None
    x, y = pos["x"], pos["y"]
    for name, (px, py) in (("dragon_pit", DRAGON_PIT), ("baron_pit", BARON_PIT)):
        if (x - px) ** 2 + (y - py) ** 2 <= 1300 ** 2:
            return name
    u, v = x / MAP_SIZE, y / MAP_SIZE
    if u < 0.25 and v < 0.25:
        return "blue_base"
    if u > 0.75 and v > 0.75:
        return "red_base"
    if (u < 0.15 and v > 0.25) or (v > 0.85 and u < 0.75):
        return "top_lane"
    if (v < 0.15 and u > 0.25) or (u > 0.85 and v < 0.75):
        return "bot_lane"
    if abs(u - v) < 0.08:
        return "mid_lane"
    if abs(u + v - 1) < 0.1:
        return "river"
    return "blue_jungle" if u + v < 1 else "red_jungle"


def _same_object(e: dict, item: dict) -> bool:
    """우리 오브젝트 이벤트 종류(subtype: grubs · dragon:water · baron …)가 op.gg 기록과 맞는지. 종류를 모르면 맞다고 본다."""
    sub = e.get("subtype")
    if not sub:
        return True
    kind, _, detail = sub.partition(":")
    return kind == item.get("type") and (not detail or not item.get("subtype") or detail == item["subtype"])


def _match(item: dict, events: list[dict], kind: str) -> tuple[int | None, float | None]:
    """같은 진영(팀을 모르는 이벤트는 종류만)·종류의 우리 이벤트 중 시각이 가장 가까운 것 (MATCH_SEC 안)."""
    best = None
    for e in events:
        if e["type"] != kind or abs(e["t"] - item["video_sec"]) > MATCH_SEC:
            continue
        if kind == "objective" and not _same_object(e, item):
            continue
        if e["team"] in (item.get("side"), None):
            d = round(e["t"] - item["video_sec"], 2)
            if best is None or abs(d) < abs(best[1]):
                best = (e["id"], d)
    return best or (None, None)


def _reference_items(ref: dict | None, clock: list[dict], players: dict, events: list[dict],
                     start: float, end: float) -> tuple[list[dict], list[dict]]:
    """op.gg 기록 → (장면에 넣을 항목, 분석 범위 밖 항목). 항목마다 video_sec 와 읽기 쉬운 값을 붙인다."""
    if not ref:
        return [], []
    items = []
    for k in ref.get("kills", []):
        items.append({"type": "kill", **k, **{f"killer_{a}": b for a, b in _who(players, k.get("killer")).items()},
                      "area": map_area(k.get("position"))})
    for o in ref.get("objectives", []):
        items.append({**o, "type": o["type"], "object": True, "area": map_area(o.get("position")),
                      **{f"killer_{a}": b for a, b in _who(players, o.get("killer")).items()}})
    minutes = sorted({round(f[0]) for fr in (ref.get("positions") or {}).values() for f in fr})
    for m in minutes:
        pos = {slot: fr_xy for slot, fr in ref["positions"].items() for t, *fr_xy in fr if round(t) == m}
        gold = {side: g for side, fr in (ref.get("gold") or {}).items() for t, g in fr if round(t) == m}
        items.append({"type": "minute", "t": float(m), "game_time": f"{m // 60:02d}:{m % 60:02d}",
                      "positions": {s: {"x": p[0], "y": p[1], "area": map_area({"x": p[0], "y": p[1]})}
                                    for s, p in pos.items()}, "gold": gold})
    inside, outside = [], []
    for it in items:
        v = to_video(clock, it["t"])
        it["video_sec"] = round(v, 3) if v is not None else None
        if it.pop("object", False) or it["type"] == "kill":
            kind = "kill" if it["type"] == "kill" else "objective"
            it["matched_event"], it["diff_sec"] = _match(it, events, kind) if v is not None else (None, None)
        (inside if v is not None and start <= v < end else outside).append(it)
    return inside, outside


FILL_KINDS = ("replay",)  # 이 장면 종류에서 짝이 없는 op.gg 기록만 채운다 (경기 화면에서 놓친 건 놓침으로 남김)


def _fill_from_reference(refs: list[dict], nodes: list[dict], lines: dict[int, dict]) -> list[dict]:
    """짝이 없는 op.gg 킬·오브젝트 중 리플레이 장면(점수판이 안 보임)에 떨어진 것을 이벤트로 만든다."""
    out = []
    for i, r in enumerate(refs, 1):
        if r["type"] == "minute" or r.get("matched_event") is not None:
            continue
        cut = next((n for n in reversed(nodes) if n["fields"]["from_sec"] <= r["video_sec"]), nodes[0])
        if cut["fields"]["kind"] not in FILL_KINDS:
            continue
        t = r["video_sec"]
        is_kill = r["type"] == "kill"
        sub = None if is_kill else (f"dragon:{r['subtype']}" if r["type"] == "dragon" and r.get("subtype") else r["type"])
        out.append({"id": f"opgg-{i}", "t": t, "game_time": r["game_time"], "type": "kill" if is_kill else "objective",
                    "subtype": sub, "team": r.get("side"), "killer": r.get("killer"), "killer_name": r.get("killer_name"),
                    "killer_champion": r.get("killer_champion"), "victims": [], "assists": [], "tags": [],
                    "source": ["opgg"], "filled": True,
                    "lines": [{k: v for k, v in ln.items() if k in ("id", "from_sec", "text")}
                              for ln in lines.values() if ln["to_sec"] >= t - 5 and ln["from_sec"] <= t + 5]})
        r["filled_event"] = out[-1]["id"]
    return out


def _signals(signals: dict, a: float, b: float) -> dict:
    """장면 [a, b) 의 초 단위 신호와 요약."""
    step, s0 = signals.get("step_sec") or 1.0, signals.get("start_sec", 0.0)
    i, j = max(0, int((a - s0) // step)), max(0, int((b - s0) // step) + 1)
    out = {k: (signals.get(k) or [])[i:j] for k in ("loudness", "gold_diff", "cut_rate")}
    loud = [x for x in out["loudness"] if x is not None]
    out["loudness_max"] = max(loud) if loud else None
    out["loudness_mean"] = round(sum(loud) / len(loud), 3) if loud else None
    return out
