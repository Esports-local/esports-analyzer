"""analyzer.json: 단계 결과를 시간순 트리 하나로 묶는다 (sevenmin_master 의 mastering_data.json 방식).

run_pipeline 이 끝날 때마다 <outputs>/<영상>/analyzer.json 을 다시 만든다. 단계 파일(work/*.json)은 그대로 둔다.

    {schema_version,
     header:    {video_id, video_type, range, context(header 단계 결과), segment_types},
     segments:  [phase{fields, children: [cut{fields: {from_sec, to_sec, kind, transition_in, game_time,
                                                         events, lines, screen}}]}]   시간순
     signals:   context 의 초 단위 신호 (무거워서 header 밖)
     reference: op.gg 세트 기록 (검증 정답지, 경기 시각 → video_sec 를 붙임)
     raw:       {roi_ocr}  화면 OCR 원본 구간 (화면 OCR 탭·검사 화면용)
     run_log}

phase: 컷들을 큰 단위로 묶은 것. 경기 구간(context.game) 안의 컷은 모두 game(리플레이·캠 포함),
밖은 화면 종류(draft · postgame · countdown · break)가 같은 컷끼리.
cut: 장면 전환 사이 구간 (segments). 이벤트·대사는 시각(대사는 시작 시각)이 속한 컷에 들어간다.
전환 구간(컷과 컷 사이)에 떨어진 것은 바로 앞 컷에 넣는다.
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
            "events": {"type": "list[object]", "description": "킬·오브젝트·타워 (context.events)"},
            "lines": {"type": "list[object]", "description": "해설 대사 {id, from_sec, to_sec, text, raw?, fixes?}"},
            "screen": {"type": "object", "description": "장면 끝 화면 값 {kills: {blue, red}, gold_diff}"},
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
    asr, roi, ref = _load(work, "asr") or {}, _load(work, "roi_ocr"), _load(work, "opgg_timeline")
    start = pre.get("range_start_sec") or 0.0
    end = pre.get("duration_sec") or max((s["end_sec"] for s in ctx.get("segments", [])), default=start)

    cuts = [dict(s) for s in ctx.get("segments", [])] or [
        {"id": 1, "start_sec": start, "end_sec": end, "kind": None, "transition_in": None, "game_time": None}]
    cut_nodes = [_cut(c) for c in cuts]
    _attach(cut_nodes, ctx.get("events", []), "events", lambda e: e["t"])
    _attach(cut_nodes, [_line(ln) for ln in asr.get("lines", [])], "lines", lambda ln: ln["from_sec"])
    for node in cut_nodes:
        node["fields"]["screen"] = _screen(roi, ctx.get("signals", {}), node["fields"]["to_sec"])

    log_path = work / "run_log.json"
    return {
        "schema_version": 1,
        "header": {"video_id": work_dir.name, "video_type": "lol", "range": {"from_sec": start, "to_sec": end},
                   "context": header or {}, "game": ctx.get("game", {}), "players": ctx.get("players", {}),
                   "clock": ctx.get("clock", []), "segment_types": SEGMENT_TYPES},
        "segments": _phases(cut_nodes, ctx.get("game") or {}),
        "signals": ctx.get("signals", {}),
        "reference": _reference(ref, ctx.get("clock", [])),
        "raw": {"roi_ocr": roi},
        "run_log": json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else [],
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


def _reference(ref: dict | None, clock: list[dict]) -> dict | None:
    """op.gg 기록에 video_sec(화면 경기 시간으로 옮긴 영상 시각, 분석 범위 밖이면 None)를 붙인다."""
    if not ref:
        return None
    out = dict(ref)
    for key in ("kills", "objectives"):
        out[key] = [{**x, "video_sec": (round(v, 3) if (v := to_video(clock, x["t"])) is not None else None)}
                    for x in ref.get(key, [])]
    return out
