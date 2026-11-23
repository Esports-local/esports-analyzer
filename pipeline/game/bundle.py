"""analyzer.json: 단계 결과를 시간순 트리 하나로 묶는다 (sevenmin_master 의 mastering_data.json 방식).

run_pipeline 이 끝날 때마다 <outputs>/<영상>/analyzer.json 을 다시 만든다. 단계 파일(work/*.json)은 그대로 둔다.

    {schema_version,
     header:   {video_id, video_type, range, context(header 단계 결과), game, players, clock, segment_types},
     segments: [phase{fields, children: [cut{fields}]}]  시간순. 그 시간의 데이터는 전부 cut 안에 있다}
header.sync: 시계 동기화 검증 (timeline.sync: op.gg 킬 ↔ 화면 팀 킬 수 변화 차이), header.event_source: opgg · ocr

cut.fields: from_sec · to_sec · kind · transition_in · game_time
    events     경기 이벤트 (timeline → context. op.gg 기록이 있으면 킬·오브젝트는 op.gg, 타워는 화면).
               선수 이름·챔피언, op.gg 위치가 있으면 맵 구역(area)
    lines      해설 대사 (교정 원문 raw · 내역 fixes)
    signals    장면 길이만큼의 초 단위 소리 크기·골드 차이·컷 빈도 + 요약
    state      장면 끝 경기 상태 {kills: 이벤트 누적 팀 킬, gold_diff: 화면 값 (없으면 op.gg 1분 골드 보간), gold_source}
화면 OCR 원본은 넣지 않는다 (work/roi_ocr.json). op.gg 기록 원본도 넣지 않는다 (work/opgg_timeline.json).

phase: 경기 구간(context.game) 안의 컷은 모두 game(리플레이·캠 포함), 밖은 화면 종류가 같은 컷끼리.
시각이 장면 사이(전환 구간)면 바로 앞 장면, 첫 장면 앞이면 첫 장면에 넣는다. 대사는 시작 시각 기준.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from pipeline.game.context import video_to_game

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
            "events": {"type": "list[object]", "description": "킬·오브젝트·타워 (context.events, op.gg 기록 + 화면 타워), 앞뒤 해설 대사(lines)·맵 구역(area) 포함"},
            "lines": {"type": "list[object]", "description": "해설 대사 {id, from_sec, to_sec, text, raw?, fixes?}"},
            "state": {"type": "object", "description": "장면 끝 경기 상태 {kills: {blue, red} 이벤트 누적, gold_diff 블루 기준 (화면 값, 없으면 op.gg 1분 보간), gold_source screen · opgg}"},
            "signals": {"type": "object", "description": "장면 길이만큼 step_sec 간격 loudness(소리 크기 0~1)·gold_diff·cut_rate (첫 값 시각 start_sec) + loudness_max·loudness_mean"},

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


def build(work_dir: Path) -> dict:
    work = work_dir / "work"
    header, pre, ctx = _load(work, "header"), _load(work, "preprocess") or {}, _load(work, "context") or {}
    asr, ref, tl = _load(work, "asr") or {}, _load(work, "opgg_timeline"), _load(work, "timeline") or {}
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
    for node in nodes:
        f = node["fields"]
        f["signals"] = _signals(signals, f["from_sec"], f["to_sec"])
        f["state"] = _state(events, ref, clock, signals, f["to_sec"])

    return {
        "schema_version": 1,
        "header": {"video_id": work_dir.name, "video_type": "lol", "range": {"from_sec": start, "to_sec": end},
                   "context": header or {}, "game": ctx.get("game", {}), "players": players, "clock": clock,
                   "event_source": tl.get("source"), "sync": tl.get("sync") or {}, "segment_types": SEGMENT_TYPES},
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
                       "events": [], "lines": []},
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


def _state(events: list[dict], ref: dict | None, clock: list[dict], signals: dict, t: float) -> dict:
    """t 초(컷 끝) 경기 상태. 팀 킬은 op.gg 킬 누적(분석 범위 앞 킬 포함, op.gg 가 없으면 이벤트 누적),
    골드 차이는 화면 값 (리플레이 등으로 없으면 op.gg 1분 값 보간)."""
    kills = {"blue": 0, "red": 0}
    g = game_time_at(clock, t)
    if ref and ref.get("kills") and g is not None:
        for k in ref["kills"]:
            if k["t"] < g and k.get("side") in kills:
                kills[k["side"]] += 1
    else:
        for e in events:
            if e["type"] == "kill" and e["t"] < t and e.get("team") in kills:
                kills[e["team"]] += 1
    arr = signals.get("gold_diff") or []
    i = int((t - signals.get("start_sec", 0.0)) // (signals.get("step_sec") or 1.0)) - 1
    screen = arr[i] if 0 <= i < len(arr) else None
    if screen is not None:  # 화면 값 먼저: 1초 단위라 킬 순간 변화가 잡힌다 (op.gg 1분 보간은 싸움을 평평하게 편다)
        return {"kills": kills, "gold_diff": screen, "gold_source": "screen"}
    return {"kills": kills, "gold_diff": _gold_diff(ref, g), "gold_source": "opgg"}


def game_time_at(clock: list[dict], t: float) -> float | None:
    """영상 t 초 → 경기 초. 시계 구간 밖(리플레이 등)이면 바로 앞 시계 구간 차이로 (경기는 계속 흐른다)."""
    g = video_to_game(clock, t)
    if g is not None:
        return g
    prev = [c for c in clock if c["start_sec"] <= t]
    return t - prev[-1]["start_sec"] + prev[-1]["game_start_sec"] if prev else None


def _gold_diff(ref: dict | None, g: float | None) -> float | None:
    """op.gg 1분 골드(팀별 누적)를 경기 g 초로 선형 보간한 블루 - 레드."""
    if not ref or g is None:
        return None
    vals = {}
    for side in ("blue", "red"):
        pts = sorted((ref.get("gold") or {}).get(side) or [])
        if len(pts) < 2 or not pts[0][0] <= g <= pts[-1][0]:
            return None
        for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
            if t0 <= g <= t1:
                vals[side] = v0 + (v1 - v0) * ((g - t0) / (t1 - t0) if t1 > t0 else 0)
                break
    return round(vals["blue"] - vals["red"]) if len(vals) == 2 else None


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
    if e.get("position"):
        out["area"] = map_area(e["position"])
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


def _signals(signals: dict, a: float, b: float) -> dict:
    """장면 [a, b) 의 초 단위 신호와 요약."""
    step, s0 = signals.get("step_sec") or 1.0, signals.get("start_sec", 0.0)
    i, j = max(0, int((a - s0) // step)), max(0, int((b - s0) // step) + 1)
    out = {"start_sec": round(s0 + i * step, 3), "step_sec": step}  # 배열 첫 값의 시각·간격
    out.update({k: (signals.get(k) or [])[i:j] for k in ("loudness", "gold_diff", "cut_rate")})
    loud = [x for x in out["loudness"] if x is not None]
    out["loudness_max"] = max(loud) if loud else None
    out["loudness_mean"] = round(sum(loud) / len(loud), 3) if loud else None
    return out
