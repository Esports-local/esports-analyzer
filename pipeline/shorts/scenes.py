"""숏폼 후보: analyzer.json 의 컷을 대사로 이어 장면으로 묶고, 이벤트가 있는 장면을 후보로 고른다 (LLM 없음).

묶기: 컷 경계를 넘어가는 대사 문장이 있으면 앞뒤 컷을 같은 장면으로 본다 (해설이 이어지는 동안은 같은 장면).
화면 종류(live·replay)가 바뀌면 문장이 이어져도 나눈다.
후보: live 장면 중 킬이나 오브젝트(유충 제외)가 있는 것. 유충·타워만 있는 장면은 뺀다.
화면 종류 규칙이 없는 레이아웃(LCK 밖)은 kind 가 비어 있으므로, 비어 있으면 경기 구간(header.game) 안의 장면을 live 로 본다.
길이는 자르지 않는다 (장면 처음~끝 그대로, 11-23 결과 보고 정함).
이벤트별 클립(clips, 모음형 숏폼 재료): 이벤트가 든 장면 안에서, 이벤트 시각부터 앞뒤 해설 문장으로 넓히되
소리 크기(loud)가 clip_loud 이상인 문장까지만 (싸움 반응). 이벤트 clip_min_pre~clip_max_pre 초 앞,
clip_min_post~clip_max_post 초 뒤 안. 이벤트 뒤 replay_within 초 안에 시작하는 리플레이 장면도 붙여 둔다.

파이프라인 단계 shorts: context 다음. 컷 트리는 analyzer.json 과 같은 것(bundle.build)을 그때 만들어 쓴다
(analyzer.json 은 파이프라인이 끝날 때 쓰므로 단계 중에는 낡았을 수 있다). 결과 work/shorts.json.

표 보기: python -m pipeline.shorts.scenes outputs/<영상>   (analyzer.json 으로 장면·후보 표를 출력)
"""

import json
import logging
import sys
from pathlib import Path

from pipeline.io import save_stage, stage_path
from pipeline.schema import ShortCandidate, ShortClip, ShortsResult

log = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "cross_tol_sec": 0.3,           # 문장이 경계를 이 초보다 더 넘어야 이어진 것으로 본다 (받아쓰기 시각 오차)
    "clip_loud": 0.85,              # 이 소리 크기 이상인 문장까지 클립을 넓힌다
    "clip_min_pre": 4.0, "clip_max_pre": 12.0,
    "clip_min_post": 3.0, "clip_max_post": 8.0,
    "replay_within": 90.0,          # 이벤트 뒤 이 초 안에 시작하는 리플레이 장면을 붙인다
    "use_llm": True,                # 후보 순위·제목·자를 위치를 LLM 으로 (키가 없으면 규칙)
    "llm_model": "gemini-3.8-flash",
}
CROSS_TOL = DEFAULT_PARAMS["cross_tol_sec"]
MINOR_SUBTYPES = ("grubs",)         # 후보로 치지 않는 오브젝트
MINOR_TYPES = ("tower",)


def cuts_of(data: dict) -> list[dict]:
    """analyzer.json 의 모든 phase 아래 컷을 시간순으로."""
    cuts = [c for p in data.get("segments", []) for c in p.get("children", []) if c.get("type") == "cut"]
    return sorted(cuts, key=lambda c: c["fields"]["from_sec"])


def run(video: Path, work_dir: Path, params: dict | None = None) -> Path:
    """work/shorts.json 을 만든다. 영상은 읽지 않는다."""
    from pipeline.game import bundle

    p = {**DEFAULT_PARAMS, **(params or {})}
    if not stage_path(work_dir, "context").exists():
        raise FileNotFoundError("shorts 는 work/context.json 이 필요합니다 (context 단계 먼저)")
    data = bundle.build(work_dir)
    scenes = group(cuts_of(data), p["cross_tol_sec"])
    cands = candidates(data, p["cross_tol_sec"])
    clips = event_clips(data, p, _load_ref(work_dir))
    cands = _with_llm(work_dir, data, cands, p)
    result = ShortsResult(scenes=len(scenes), candidates=[ShortCandidate(**c) for c in cands],
                          clips=[ShortClip(**c) for c in clips])
    if cuts_of(data) and all(c["fields"].get("kind") is None for c in cuts_of(data)):
        log.warning("shorts: 화면 종류(kind)가 없어 경기 구간 안 장면을 모두 live 로 봅니다 (리플레이도 섞일 수 있음)")
    log.info("shorts: 컷 %d → 장면 %d → 후보 %d, 이벤트 클립 %d", len(cuts_of(data)), len(scenes), len(cands), len(clips))
    return save_stage(work_dir, "shorts", result, p)


def _with_llm(work_dir: Path, data: dict, cands: list[dict], p: dict) -> list[dict]:
    """LLM 순위·제목·자를 위치(검사 후 최종값)를 후보에 붙인다. 기록은 work/shorts_llm.json, 입력은 shorts_llm_input.json."""
    from pipeline.shorts import llm_input, rank

    if not cands:
        return cands
    inp = llm_input.build(data, cands)
    work = work_dir / "work"
    (work / "shorts_llm_input.json").write_text(json.dumps(inp, ensure_ascii=False, indent=1), encoding="utf-8")
    prev_path = work / "shorts_llm.json"
    previous = json.loads(prev_path.read_text(encoding="utf-8")) if prev_path.exists() else None
    res = rank.run(inp, previous, model=p["llm_model"], allow_call=p["use_llm"])
    res["input"] = "work/shorts_llm_input.json"
    prev_path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    if res["error"]:
        log.warning("shorts: %s → 규칙 순서·자동 제목", res["error"])
    else:
        log.info("shorts: LLM %s%s, 토큰 %s", res["model"], " (지난 응답 재사용)" if res["reused"] else "",
                 {k: v for k, v in res["usage"].items() if k.endswith("TokenCount")})
    final = {c["id"]: c["final"] for c in res["candidates"]}
    out = []
    for c in cands:
        f = final.get(c["id"]) or {}
        out.append({**c, "rank": f.get("rank"), "title": f.get("title"), "titles": f.get("titles") or [],
                    "reason": f.get("reason") or "", "clip_start_sec": f.get("start_sec"),
                    "clip_end_sec": f.get("end_sec"), "source": f.get("source"), "warnings": f.get("warnings") or []})
    return sorted(out, key=lambda c: c["rank"] or 999)


def group(cuts: list[dict], tol: float = CROSS_TOL) -> list[list[dict]]:
    """컷들을 장면(컷 묶음)으로. 경계를 넘는 문장이 있고 화면 종류가 같으면 잇는다."""
    if not cuts:
        return []
    lines = {l["id"]: l for c in cuts for l in c["fields"].get("lines", [])}
    spans = [(l["from_sec"], l.get("to_sec", l["from_sec"])) for l in lines.values()]
    scenes = [[cuts[0]]]
    for a, b in zip(cuts, cuts[1:]):
        t = a["fields"]["to_sec"]
        crossed = any(s < t - tol and e > t + tol for s, e in spans)
        if crossed and a["fields"].get("kind") == b["fields"].get("kind"):
            scenes[-1].append(b)
        else:
            scenes.append([b])
    return scenes


def is_major(e: dict) -> bool:
    if e.get("type") in MINOR_TYPES:
        return False
    if e.get("type") == "objective" and (e.get("subtype") or "").split(":")[0] in MINOR_SUBTYPES:
        return False
    return e.get("type") in ("kill", "objective")


def scene_info(scene: list[dict]) -> dict:
    f0, f1 = scene[0]["fields"], scene[-1]["fields"]
    events = [e for c in scene for e in c["fields"].get("events", [])]
    lines = {l["id"]: l for c in scene for l in c["fields"].get("lines", [])}
    return {
        "cuts": [c["id"] for c in scene],
        "kind": f0.get("kind"),
        "start_sec": f0["from_sec"], "end_sec": f1["to_sec"],
        "duration_sec": round(f1["to_sec"] - f0["from_sec"], 2),
        "events": [{"id": e["id"], "t": e["t"], "type": e["type"], "subtype": e.get("subtype"),
                    "team": e.get("team"), "tags": e.get("tags", [])} for e in events],
        "lines": [l["text"] for l in sorted(lines.values(), key=lambda l: l["from_sec"])],
    }


def candidates(data: dict, tol: float = CROSS_TOL) -> list[dict]:
    """후보 장면 목록 (시간순). id 는 c1, c2 …"""
    game = (data.get("header") or {}).get("game") or {}
    out = []
    for scene in group(cuts_of(data), tol):
        info = scene_info(scene)
        if is_live(info, game) and any(is_major(e) for e in info["events"]):
            out.append({"id": f"c{len(out) + 1}", **info})
    return out


def is_live(info: dict, game: dict) -> bool:
    """live 장면인지. kind 가 없으면(화면 종류 규칙 없는 레이아웃) 경기 구간과 겹치면 live 로 본다."""
    if info["kind"] is not None:
        return info["kind"] == "live"
    start, end = game.get("start_sec"), game.get("end_sec")
    if start is None or end is None:
        return True
    return info["start_sec"] < end and info["end_sec"] > start


def _load_ref(work_dir: Path) -> dict | None:
    path = stage_path(work_dir, "opgg_timeline")
    if not path.exists():
        return None
    ref = json.loads(path.read_text(encoding="utf-8")).get("result") or {}
    return None if ref.get("error") else ref


def kill_score(data: dict, ref: dict | None, t: float) -> dict | None:
    """영상 t 초 킬 직후 팀 킬 점수. op.gg 킬을 경기 시각으로 누적 (분석 범위 앞 킬 포함). 없으면 None."""
    from pipeline.game.bundle import game_time_at

    g = game_time_at((data.get("header") or {}).get("clock") or [], t)
    if not ref or not ref.get("kills") or g is None:
        return None
    score = {"blue": 0, "red": 0}
    for k in ref["kills"]:
        if k["t"] <= g + 0.5 and k.get("side") in score:
            score[k["side"]] += 1
    return score


def event_clips(data: dict, p: dict | None = None, ref: dict | None = None) -> list[dict]:
    """이벤트마다 해설 기준 클립 (시간순). ref(op.gg 기록)가 있으면 킬마다 직후 점수."""
    from pipeline.shorts.llm_input import _loud, _loud_samples

    p = {**DEFAULT_PARAMS, **(p or {})}
    s0 = ((data.get("header") or {}).get("range") or {}).get("from_sec") or 0.0
    scenes = group(cuts_of(data), p["cross_tol_sec"])
    cand_of = {tuple(c["cuts"]): c["id"] for c in candidates(data, p["cross_tol_sec"])}
    out = []
    for i, scene in enumerate(scenes):
        a, b = scene[0]["fields"]["from_sec"], scene[-1]["fields"]["to_sec"]
        samples = _loud_samples(scene, s0)
        lines = sorted({l["id"]: l for c in scene for l in c["fields"].get("lines", [])}.values(),
                       key=lambda l: l["from_sec"])
        for l in lines:
            l["_loud"] = _loud(samples, l) or 0.0
        replay = _replay_after(scenes, i, p["replay_within"])
        for c in scene:
            for e in c["fields"].get("events", []):
                start, end = clip_span(e["t"], lines, a, b, p)
                out.append({"event_id": e["id"], "t": e["t"], "game_time": e.get("game_time"), "type": e["type"],
                            "subtype": e.get("subtype"), "team": e.get("team"), "killer": e.get("killer"),
                            "victims": e.get("victims") or [], "assists": e.get("assists") or [],
                            "tags": e.get("tags") or [], "area": e.get("area"),
                            "start_sec": start, "end_sec": end, "on_screen": c["fields"].get("kind"),
                            "scene": cand_of.get(tuple(x["id"] for x in scene)),
                            "replay": replay if replay and replay["start_sec"] - e["t"] <= p["replay_within"] else None,
                            "score": kill_score(data, ref, e["t"]) if e["type"] == "kill" else None})
    return sorted(out, key=lambda x: x["t"])


def clip_span(t: float, lines: list[dict], a: float, b: float, p: dict) -> tuple[float, float]:
    """이벤트 t 의 클립 [시작, 끝]. 문장(_loud 붙은 것)은 시간순, [a, b] 는 장면 범위."""
    lo, hi = max(a, t - p["clip_max_pre"]), min(b, t + p["clip_max_post"])
    start, end = max(a, t - p["clip_min_pre"]), min(b, t + p["clip_min_post"])
    before = [l for l in lines if l["from_sec"] < t]
    for l in reversed(before):  # 이벤트 앞: 큰 소리 문장이 이어지는 동안 넓힘
        if l["from_sec"] < lo or l["_loud"] < p["clip_loud"]:
            break
        start = min(start, l["from_sec"])
    for l in (l for l in lines if l["to_sec"] > t):  # 이벤트 뒤
        if l["to_sec"] > hi or l["_loud"] < p["clip_loud"]:
            break
        end = max(end, l["to_sec"])
    # 문장 중간에서 시작·끝나지 않게 걸친 문장 경계로 (범위 안일 때만)
    for l in lines:
        if l["from_sec"] < start < l["to_sec"] and l["from_sec"] >= lo:
            start = l["from_sec"]
        if l["from_sec"] < end < l["to_sec"] and l["to_sec"] <= hi:
            end = l["to_sec"]
    return round(start, 3), round(end, 3)


def _replay_after(scenes: list[list[dict]], i: int, within: float) -> dict | None:
    """i 번째 장면 뒤 첫 리플레이 장면 (within 초 안에 시작하는 것)."""
    end = scenes[i][-1]["fields"]["to_sec"]
    for s in scenes[i + 1:]:
        f0 = s[0]["fields"]
        if f0["from_sec"] - end > within:
            return None
        if f0.get("kind") == "replay":
            return {"start_sec": f0["from_sec"], "end_sec": s[-1]["fields"]["to_sec"], "cuts": [c["id"] for c in s]}
    return None


def _ev_label(e: dict) -> str:
    name = {"kill": "킬", "tower": "타워"}.get(e["type"]) or (e.get("subtype") or e["type"])
    tags = f"({','.join(e['tags'])})" if e.get("tags") else ""
    return f"{name}{tags}@{e['t']:.0f}"


def main(argv: list[str]) -> None:
    out_dir = Path(argv[0])
    data = json.loads((out_dir / "analyzer.json").read_text(encoding="utf-8"))
    scenes = group(cuts_of(data))
    cands = {tuple(c["cuts"]) for c in candidates(data)}
    print(f"컷 {len(cuts_of(data))}개 → 장면 {len(scenes)}개 → 후보 {len(cands)}개\n")
    for scene in scenes:
        s = scene_info(scene)
        mark = "★" if tuple(s["cuts"]) in cands else " "
        rng = s["cuts"][0] if len(s["cuts"]) == 1 else f"{s['cuts'][0]}~{s['cuts'][-1]}"
        evs = ", ".join(_ev_label(e) for e in s["events"])
        print(f"{mark} {rng:<14} {len(s['cuts']):>2}컷 {s['start_sec']:7.1f}~{s['end_sec']:7.1f} "
              f"{s['duration_sec']:5.1f}초 {s['kind'] or '-':<7} {evs}")


if __name__ == "__main__":
    main(sys.argv[1:])
