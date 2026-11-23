"""숏폼 후보: analyzer.json 의 컷을 대사로 이어 장면으로 묶고, 이벤트가 있는 장면을 후보로 고른다 (LLM 없음).

묶기: 컷 경계를 넘어가는 대사 문장이 있으면 앞뒤 컷을 같은 장면으로 본다 (해설이 이어지는 동안은 같은 장면).
화면 종류(live·replay)가 바뀌면 문장이 이어져도 나눈다.
후보: live 장면 중 킬이나 오브젝트(유충 제외)가 있는 것. 유충·타워만 있는 장면은 뺀다.
화면 종류 규칙이 없는 레이아웃(LCK 밖)은 kind 가 비어 있으므로, 비어 있으면 경기 구간(header.game) 안의 장면을 live 로 본다.
길이는 자르지 않는다 (장면 처음~끝 그대로, 11-23 결과 보고 정함).

파이프라인 단계 shorts: context 다음. 컷 트리는 analyzer.json 과 같은 것(bundle.build)을 그때 만들어 쓴다
(analyzer.json 은 파이프라인이 끝날 때 쓰므로 단계 중에는 낡았을 수 있다). 결과 work/shorts.json.

표 보기: python -m pipeline.shorts.scenes outputs/<영상>   (analyzer.json 으로 장면·후보 표를 출력)
"""

import json
import logging
import sys
from pathlib import Path

from pipeline.io import save_stage, stage_path
from pipeline.schema import ShortCandidate, ShortsResult

log = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "cross_tol_sec": 0.3,           # 문장이 경계를 이 초보다 더 넘어야 이어진 것으로 본다 (받아쓰기 시각 오차)
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
    result = ShortsResult(scenes=len(scenes), candidates=[ShortCandidate(**c) for c in cands])
    if cuts_of(data) and all(c["fields"].get("kind") is None for c in cuts_of(data)):
        log.warning("shorts: 화면 종류(kind)가 없어 경기 구간 안 장면을 모두 live 로 봅니다 (리플레이도 섞일 수 있음)")
    log.info("shorts: 컷 %d → 장면 %d → 후보 %d", len(cuts_of(data)), len(scenes), len(cands))
    return save_stage(work_dir, "shorts", result, p)


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
