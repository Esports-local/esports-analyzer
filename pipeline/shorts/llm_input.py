"""LLM 입력: analyzer.json 을 같은 모양(header + segments 트리)으로 두고 숏폼 후보 장면만 남겨 줄인다.

- header: 경기 정보(줄임) · players(슬롯 → "팀 이름 (챔피언)") · segment_types(scene·cut 필드 설명)
- segments: 후보마다 scene 노드 하나 → 그 장면의 cut 들
  - cut: from_sec · to_sec · game_time · events(이름 붙인 것) · lines(문장마다 소리 크기 loud) · state
  - 뺀 것: reference(op.gg 정답지) · 1초 단위 signals 배열 · transition_in · kind(후보는 모두 live) · 대사 raw·fixes
- 시각은 analyzer.json 과 같이 원본 영상 기준 초. LLM 은 자를 위치를 lines.id 로 답한다

보기: python -m pipeline.shorts.llm_input outputs/<영상>   (work/ 로 트리·후보를 그때 만들어 결과를 출력)
"""

import json
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

SEGMENT_TYPES = {
    "scene": {
        "description": "숏폼 후보 장면. 해설이 이어지는 컷들을 묶은 것이고 킬·오브젝트가 하나 이상 있다",
        "fields": {
            "from_sec": "장면 시작 (원본 영상 초)", "to_sec": "장면 끝", "duration_sec": "길이",
            "score": "팀 킬 점수 {start: {blue, red}, end: {blue, red}} (장면 직전·장면 끝)",
            "gold_diff": "골드 차이 {start, end}, 블루 기준 (양수면 블루가 앞섬, 장면 직전·장면 끝)",
        },
        "allowed_children": ["cut"],
    },
    "cut": {
        "description": "장면 전환 사이의 연속 화면 (카메라가 바뀌면 새 cut)",
        "fields": {
            "from_sec": "시작", "to_sec": "끝", "game_time": "화면 경기 시간 [처음, 끝]",
            "events": "킬·오브젝트(op.gg 기록, t 는 실제 일어난 시각)·타워(화면 숫자가 바뀐 시각이라 1~2초 늦다). "
                      "area 는 맵 구역 (top_lane · mid_lane · bot_lane · river · blue_jungle · dragon_pit …). "
                      "킬당한 선수·도움은 화면 KDA 로 찾아 비어 있을 수 있다",
            "lines": "해설 받아쓰기 {id, from_sec, to_sec, text, loud}. 오타·잘못 들은 이름이 많다. "
                     "loud 는 그 문장 동안 소리 크기 최댓값 0~1 (외침일수록 큼)",
            "state": "cut 끝 경기 상태 {kills: {blue, red}, gold_diff, gold_source}",
        },
        "allowed_children": [],
    },
}


def build(data: dict, candidates: list[dict]) -> dict:
    """analyzer.json(data) 과 후보 목록(scenes.candidates 결과) → LLM 입력."""
    header = data.get("header", {})
    players = header.get("players", {})
    cuts = [c for p in data.get("segments", []) for c in p.get("children", []) if c.get("type") == "cut"]
    cuts.sort(key=lambda c: c["fields"]["from_sec"])
    index = {c["id"]: i for i, c in enumerate(cuts)}
    s0 = (header.get("range") or {}).get("from_sec") or 0.0  # 옛 analyzer.json(signals 에 start_sec 없음)용

    scenes = []
    for cand in candidates:
        missing = [cid for cid in cand["cuts"] if cid not in index]
        if missing:  # 후보와 트리가 다른 실행에서 나옴
            log.warning("후보 %s 의 컷 %s 이 트리에 없어 뺍니다", cand["id"], missing)
            continue
        mine = [cuts[index[cid]] for cid in cand["cuts"]]
        before = cuts[index[cand["cuts"][0]] - 1] if index[cand["cuts"][0]] > 0 else None
        before_state = ((before or {}).get("fields") or {}).get("state") or {}
        end_state = mine[-1]["fields"].get("state") or {}
        first_sig = mine[0]["fields"].get("signals") or {}
        loud = _loud_samples(mine, s0)
        seen: set = set()
        children = [_cut(c, players, loud, seen) for c in mine]
        scenes.append({
            "id": cand["id"], "type": "scene",
            "fields": {
                "from_sec": cand["start_sec"], "to_sec": cand["end_sec"], "duration_sec": cand["duration_sec"],
                "score": {"start": before_state.get("kills"), "end": end_state.get("kills")},
                "gold_diff": {"start": before_state.get("gold_diff", _first(first_sig.get("gold_diff"))),
                              "end": end_state.get("gold_diff")},
            },
            "children": children,
        })
    return {"header": _header(header), "segments": scenes}


def _header(h: dict) -> dict:
    ctx = h.get("context") or {}
    m = ctx.get("match") or {}
    teams = ctx.get("teams") or {}
    return {
        "match": {"name": m.get("name"), "league": m.get("league"), "stage": m.get("stage"), "year": m.get("year"),
                  "format": m.get("format"), "set": m.get("set"),
                  "series_before": (ctx.get("series") or {}).get("score_before")},
        "teams": {side: f"{t.get('name')} ({t.get('full')})" for side, t in teams.items() if t},
        "players": {slot: _who(p) for slot, p in (h.get("players") or {}).items()},
        "segment_types": SEGMENT_TYPES,
    }


def _who(p: dict | None) -> str | None:
    if not p or not p.get("name"):
        return None
    return f"{p.get('team')} {p['name']} ({p.get('champion')})" if p.get("team") else f"{p['name']} ({p.get('champion')})"


def _cut(c: dict, players: dict, loud: list[tuple[float, float, float]], seen: set) -> dict:
    f = c["fields"]
    lines = []
    for ln in f.get("lines", []):
        if ln["id"] in seen:  # 경계를 넘는 문장은 앞 cut 에만
            continue
        seen.add(ln["id"])
        lines.append({"id": ln["id"], "from_sec": ln["from_sec"], "to_sec": ln["to_sec"], "text": ln["text"],
                      "loud": _loud(loud, ln)})
    return {"id": c["id"], "type": "cut",
            "fields": {"from_sec": f["from_sec"], "to_sec": f["to_sec"], "game_time": f.get("game_time"),
                       "events": [_event(e, players) for e in f.get("events", [])],
                       "lines": lines, "state": f.get("state")},
            "children": []}


def _event(e: dict, players: dict) -> dict:
    out = {"id": e["id"], "t": e["t"], "game_time": e.get("game_time"), "type": e["type"],
           "subtype": e.get("subtype"), "team": e.get("team"),
           "killer": _who(players.get(e.get("killer"))) or e.get("killer_name"),
           "victims": [_who(players.get(s)) or s for s in e.get("victims", [])],
           "assists": [_who(players.get(s)) or s for s in e.get("assists", [])],
           "tags": e.get("tags", []), "area": e.get("area")}
    return {k: v for k, v in out.items() if v not in (None, [])}


def _loud_samples(cuts: list[dict], s0: float) -> list[tuple[float, float, float]]:
    """장면 cut 들의 소리 크기 → [(시작, 끝, 값)]. 컷 경계를 넘는 문장도 다음 컷 값까지 보게 장면 전체로 모은다."""
    out = {}
    for c in cuts:
        f = c["fields"]
        sig = f.get("signals") or {}
        step = sig.get("step_sec") or 1.0
        start = sig.get("start_sec")
        if start is None:  # 옛 analyzer.json: 1초 간격, header.range 시작 기준
            start = s0 + ((f["from_sec"] - s0) // step) * step
        for k, v in enumerate(sig.get("loudness") or []):
            if v is not None:
                t = round(start + k * step, 3)
                out[t] = (t, t + step, v)  # 경계 초는 두 컷 배열에 다 있다 (같은 값)
    return sorted(out.values())


def _loud(samples: list[tuple[float, float, float]], ln: dict) -> float | None:
    """문장 동안(겹치는 구간) 소리 크기 최댓값."""
    vals = [v for a, b, v in samples if a < ln["to_sec"] and b > ln["from_sec"]]
    return round(max(vals), 2) if vals else None


def _first(xs: list | None):
    return next((x for x in xs or [] if x is not None), None)


def main(argv: list[str]) -> None:
    from pipeline.game import bundle
    from pipeline.shorts.scenes import candidates

    out_dir = Path(argv[0])
    data = bundle.build(out_dir)  # analyzer.json·shorts.json 은 낡았을 수 있어 work/ 로 그때 만든다
    print(json.dumps(build(data, candidates(data)), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
