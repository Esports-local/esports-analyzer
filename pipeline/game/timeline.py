"""timeline 단계: 경기 이벤트(킬 · 오브젝트 · 타워)를 영상 초로 뽑는다. 숏폼은 이 이벤트를 기준점으로 구간을 잡는다.

LLM 을 쓰지 않는다. 같은 입력이면 항상 같은 결과다.

입력: work/roi_ocr.json (필수), work/opgg_timeline.json (header 단계, 있으면), work/asr.json (있으면 앞뒤 대사)

이벤트 원천 (11-23 결정: VOD 라 op.gg 기록이 이미 있으므로 화면으로 다시 만들지 않는다)
- op.gg 상세 기록이 있으면: 킬(시각·팀·킬한 선수·위치) · 오브젝트(용 종류·유충·전령·바론…)는 op.gg.
  경기 시각 → 영상 초는 화면 경기 시계(context.build_clock)로. 리플레이 동안도 경기는 흐르므로 가까운 시계 구간
  (opgg_slack_sec 안)의 차이로 옮긴다. 킬당한 선수 · 도움은 op.gg 에 없어 화면 KDA 변화로 찾는다.
- 타워는 op.gg 에 시각이 없어 항상 화면 숫자로.
- op.gg 가 없으면 (오류 · 상세 기록 없음) 아래 화면 숫자 방식으로 킬 · 오브젝트도 만든다.
- 동기화 검증(sync): op.gg 킬마다 같은 진영 화면 팀 킬 수가 오른 시각과의 차이(ocr_lag_sec). 화면 점수는 1~2초 늦게
  바뀌므로 sync_lag_ok 범위 밖이면 시계가 어긋난 것으로 outliers 에 남긴다.

화면 숫자 방식

잡음 정리 (roi_ocr 구간은 1장짜리 오독이 섞여 있다)
- min_samples 장 미만으로 이어진 구간은 버린다
- 팀 킬 · 타워 · 오브젝트 수는 줄지 않는다. 앞 값보다 작은 값은 오독으로 버린다
- 한 번에 max_jump 넘게 오르면 오독으로 보고 버린다 (같은 값이 다시 읽히면 그때 인정)

이벤트 시각
- 화면이 계속 보였으면 새 값이 처음 읽힌 때가 이벤트 시각이다
- 사이에 읽히지 않은 구간(리플레이, 캠 화면)이 있으면 그 사이 어딘가에서 일어났으므로
  start_sec(앞 값이 마지막으로 읽힌 때) ~ end_sec(새 값이 처음 읽힌 때)로 남긴다

오브젝트 종류: 레이아웃의 obj_count 항목에 subtype 이 있으면 그것 (LCK 는 grubs).
배너 (LCK): 점수판 아래 띠에 OCEAN DRAKE · BARON NASHOR 같은 글자가 뜨면 오브젝트 이벤트 (subtype dragon:water · baron …).
같은 배너가 banner_merge_sec 안에 다시 읽히면 하나로. 용은 경기 시간 옆 용 아이콘(dragon_icons, 픽셀 비율)이
배너 앞보다 뒤에 늘어난 진영이 가져간 팀. 바론·전령은 팀을 모르면 비워 둔다.

선수 (킬만): 이벤트 앞뒤 kda_window 초 안에 KDA 의 K 가 오른 같은 진영 선수 = killer,
D 가 오른 상대 진영 선수 = victims, A 가 오른 같은 진영 선수 = assists. KDA 를 못 읽었으면 비워 둔다.
"""

import json
import logging
import re
from pathlib import Path

from pipeline.game import context as ctx
from pipeline.io import save_stage, stage_path
from pipeline.schema import TimelineEvent, TimelineResult


# 팀·선수 이름처럼 경기 내내 안 바뀌는 값은 화면에 가장 오래 떠 있던 값을 쓴다 (1장짜리 잡음은 밀린다)
STATIC_TAGS = ("team", "player_name")


def static_values(runs: list[dict]) -> dict[str, dict[str, str]]:
    """{tag: {slot: 가장 오래 뜬 값}} (팀·선수 이름)."""
    dur: dict[tuple, float] = {}
    for r in runs:
        if r["tag"] in STATIC_TAGS:
            key = (r["tag"], r["slot"] or "value", r["text"])
            dur[key] = dur.get(key, 0) + r["end_sec"] - r["start_sec"]
    best: dict[tuple, tuple[str, float]] = {}
    for (tag, slot, text), d in dur.items():
        if d > best.get((tag, slot), ("", -1.0))[1]:
            best[(tag, slot)] = (text, d)
    out: dict[str, dict[str, str]] = {}
    for (tag, slot), (text, _) in best.items():
        out.setdefault(tag, {})[slot] = text
    return out

log = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "min_samples": 2,     # 이만큼 이어진 구간만 인정 (sample_fps 2 → 1초)
    "max_jump": 3,        # 한 번에 이만큼 넘게 오르면 오독
    "kda_window": 3.0,    # 킬과 KDA 변화를 짝지을 앞뒤 초
    "line_window": 5.0,   # 이벤트 앞뒤로 붙일 대사 범위(초)
    "banner_merge_sec": 30.0,  # 같은 오브젝트 배너를 하나로 묶을 범위
    "icon_before_sec": 10.0,   # 용 아이콘 비교: 배너 이만큼 앞 값과
    "icon_after_sec": 8.0,     # 배너 이만큼 뒤 값을 비교
    "use_opgg": True,          # op.gg 상세 기록이 있으면 킬·오브젝트를 op.gg 로
    "opgg_slack_sec": 60.0,    # 시계 구간 밖(리플레이·캠) op.gg 기록을 이만큼 떨어진 시계 구간까지 옮긴다
    "sync_search_sec": [-2.0, 8.0],  # op.gg 킬 시각 기준 화면 팀 킬 수 변화를 찾을 범위
    "sync_lag_ok": [-0.5, 4.0],      # 정상으로 보는 차이 (화면 - op.gg)
}
OPGG_OBJECTS = {"grubs": "grubs", "herald": "herald", "baron": "baron", "elder": "elder", "atakhan": "atakhan"}
DRAKES = {"OCEAN": "water", "INFERNAL": "fire", "MOUNTAIN": "earth", "CLOUD": "air", "HEXTECH": "hextech",
          "CHEMTECH": "chemtech"}

COUNTERS = {"team_kills": "kill", "towers": "tower", "obj_count": "objective"}
OPPONENT = {"blue": "red", "red": "blue"}
KDA_RE = re.compile(r"^(\d+)/(\d+)/(\d+)$")


def run(video: Path, work_dir: Path, params: dict | None = None) -> Path:
    """roi_ocr(+asr) 결과로 work/timeline.json 을 만든다.

    Args:
        video: 원본 영상 경로 (쓰지 않지만 단계 공통 시그니처).
        work_dir: 회차 작업 디렉터리.
        params: DEFAULT_PARAMS 덮어쓰기.

    Returns:
        저장한 work/timeline.json 경로.

    Raises:
        FileNotFoundError: roi_ocr 를 아직 돌리지 않음.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    roi = json.loads(stage_path(work_dir, "roi_ocr").read_text(encoding="utf-8"))["result"]
    lines = _load_lines(work_dir)
    ref = load_opgg(work_dir) if p["use_opgg"] else None
    if ref:
        result = build_timeline_opgg(roi["runs"], lines, ref, p)
    else:
        log.info("[timeline] op.gg 상세 기록이 없어 화면 숫자로 만듭니다")
        result = build_timeline(roi["runs"], lines, p, _subtypes(roi.get("layout")))
    log.info("[timeline] 이벤트 %d개 (%s, 원천 %s)%s", len(result.events),
             ", ".join(f"{t} {sum(e.type == t for e in result.events)}" for t in COUNTERS.values()), result.source,
             f", 영상 초로 못 옮긴 op.gg 기록 {result.unmapped}" if result.unmapped else "")
    return save_stage(work_dir, "timeline", result, p)


def load_opgg(work_dir: Path) -> dict | None:
    """op.gg 세트 기록. 없거나 오류거나 킬·오브젝트가 비었으면 None."""
    path = stage_path(work_dir, "opgg_timeline")
    if not path.exists():
        return None
    ref = json.loads(path.read_text(encoding="utf-8")).get("result") or {}
    if ref.get("error") or not ref.get("detailed") or not (ref.get("kills") or ref.get("objectives")):
        return None
    return ref


def build_timeline_opgg(runs: list[dict], lines: list[dict], ref: dict, p: dict) -> TimelineResult:
    """op.gg 킬·오브젝트(경기 시각) → 영상 초 이벤트 + 화면 숫자 타워 (파일 입출력 없음, 테스트용)."""
    cp = ctx.DEFAULT_PARAMS
    clock = ctx.build_clock(runs, cp["clock_tolerance_sec"], cp["clock_min_sec"], cp["clock_gap_sec"])
    statics = static_values(runs)
    teams, names = statics.get("team", {}), statics.get("player_name", {})
    kda = _kda_series(runs, p["min_samples"])
    span = (min((r["start_sec"] for r in runs), default=0.0), max((r["end_sec"] for r in runs), default=0.0))

    raw, unmapped, count = [], [], {}
    items = [("kill", k) for k in ref.get("kills", [])] + [("objective", o) for o in ref.get("objectives", [])]
    for etype, it in sorted(items, key=lambda x: x[1]["t"]):
        side = it.get("side")
        key = (etype, side)
        before = count.get(key, 0)
        count[key] = before + 1
        v = ctx.to_video(clock, it["t"], p["opgg_slack_sec"])
        if v is None:
            if clock and clock[0]["game_start_sec"] <= it["t"] <= clock[-1]["game_start_sec"] + clock[-1]["end_sec"] - clock[-1]["start_sec"]:
                unmapped.append(f"{etype} {it.get('game_time')}")  # 분석 범위 안인데 못 옮김
            continue
        if not span[0] <= v < span[1]:
            continue
        sub = None if etype == "kill" else _opgg_subtype(it)
        raw.append((v, etype, sub, side, before, it, [f"opgg {it.get('type', etype)} {it.get('game_time')}"]))

    events: list[TimelineEvent] = []
    for v, etype, sub, side, before, it, evidence in raw:
        ev = TimelineEvent(id=0, type=etype, subtype=sub, team=side, team_name=teams.get(side) if side else None,
                           start_sec=round(v, 3), end_sec=round(v, 3), value_before=before, value_after=before + 1,
                           game_time=it.get("game_time"), evidence=evidence, source="opgg", position=it.get("position"))
        killer = it.get("killer")
        if killer:
            ev.players["killer"] = [names.get(killer, killer)]
        if etype == "kill":
            found = _kill_players(kda, names, side, v, v, p["kda_window"])
            ev.players.update({k: found[k] for k in ("victims", "assists") if found.get(k)})
        events.append(ev)

    for slot in ("blue", "red"):  # 타워: op.gg 에 시각이 없어 화면 숫자로
        series = [r for r in runs if r["tag"] == "towers" and r["slot"] == slot]
        for before, after, start, end, evidence in counter_steps(series, p["min_samples"], p["max_jump"]):
            events.append(TimelineEvent(id=0, type="tower", team=slot, team_name=teams.get(slot), start_sec=start,
                                        end_sec=end, value_before=before, value_after=after,
                                        game_time=_game_time_at(runs, end, p["min_samples"]), evidence=evidence))

    events.sort(key=lambda e: (e.end_sec, e.team or ""))
    for i, ev in enumerate(events, 1):
        ev.id = i
        ev.line_ids = [ln["id"] for ln in lines
                       if ln["end_sec"] >= ev.start_sec - p["line_window"] and ln["start_sec"] <= ev.end_sec + p["line_window"]]
    sync = check_sync(events, runs, p)
    return TimelineResult(events=events, teams=teams, players=names, source="opgg", unmapped=unmapped, sync=sync)


def check_sync(events: list[TimelineEvent], runs: list[dict], p: dict) -> dict:
    """op.gg 킬 ↔ 화면 팀 킬 수 변화. 킬마다 ocr_lag_sec 를 채우고 요약을 돌려준다."""
    lo, hi = p["sync_search_sec"]
    ok_lo, ok_hi = p["sync_lag_ok"]
    steps = {s: [st[3] for st in counter_steps([r for r in runs if r["tag"] == "team_kills" and r["slot"] == s],
                                               p["min_samples"], p["max_jump"])] for s in ("blue", "red")}
    used: set = set()
    lags, outliers, checked = [], [], 0
    for ev in events:
        if ev.type != "kill" or ev.source != "opgg" or ev.team not in steps:
            continue
        checked += 1
        near = [(abs(t - ev.end_sec - 1.5), t) for t in steps[ev.team]
                if (ev.team, t) not in used and lo <= t - ev.end_sec <= hi]
        if not near:
            continue
        t = min(near)[1]
        used.add((ev.team, t))
        ev.ocr_lag_sec = round(t - ev.end_sec, 2)
        lags.append(ev.ocr_lag_sec)
        if not ok_lo <= ev.ocr_lag_sec <= ok_hi:
            outliers.append({"event": ev.id, "game_time": ev.game_time, "lag_sec": ev.ocr_lag_sec})
    if not lags:
        return {"checked": checked, "matched": 0}
    s = sorted(lags)
    return {"checked": checked, "matched": len(lags), "lag_median_sec": s[len(s) // 2],
            "lag_min_sec": s[0], "lag_max_sec": s[-1], "outliers": outliers}


def _opgg_subtype(o: dict) -> str | None:
    """op.gg 오브젝트 → 화면 방식과 같은 subtype (dragon:water · grubs · baron …)."""
    kind, sub = o.get("type"), o.get("subtype")
    if kind == "dragon":
        if sub == "elder":
            return "elder"
        return f"dragon:{sub}" if sub else "dragon"
    return OPGG_OBJECTS.get(kind, kind)


def _subtypes(layout: str | None) -> dict:
    """레이아웃의 카운터 영역 오브젝트 종류 {tag: subtype} (예: LCK obj_count → grubs)."""
    if not layout or not Path(layout).exists():
        return {}
    rois = json.loads(Path(layout).read_text(encoding="utf-8")).get("rois", [])
    return {r["tag"]: r["subtype"] for r in rois if r.get("subtype")}


def build_timeline(runs: list[dict], lines: list[dict], p: dict, subtypes: dict | None = None) -> TimelineResult:
    """roi_ocr 구간과 대사로 이벤트 목록을 만든다 (파일 입출력 없음, 테스트용).

    subtypes: {tag: subtype} 카운터 영역의 오브젝트 종류 (레이아웃 obj_count.subtype).
    """
    statics = static_values(runs)
    teams = statics.get("team", {})
    players = statics.get("player_name", {})
    raw = []
    for tag, etype in COUNTERS.items():
        for slot in ("blue", "red"):
            series = [r for r in runs if r["tag"] == tag and r["slot"] == slot]
            raw += [(etype, (subtypes or {}).get(tag), slot, *step)
                    for step in counter_steps(series, p["min_samples"], p["max_jump"])]
    raw += banner_events(runs, p)
    raw.sort(key=lambda x: (x[5], x[2] or ""))  # end_sec(새 값이 읽힌 때), 진영

    kda = _kda_series(runs, p["min_samples"])
    events = []
    for i, (etype, subtype, slot, before, after, start, end, evidence) in enumerate(raw, 1):
        ev = TimelineEvent(
            id=i, type=etype, subtype=subtype, team=slot, team_name=teams.get(slot) if slot else None,
            start_sec=start, end_sec=end, value_before=before, value_after=after,
            game_time=_game_time_at(runs, end, p["min_samples"]), evidence=evidence,
        )
        if etype == "kill":
            ev.players = _kill_players(kda, players, slot, start, end, p["kda_window"])
        ev.line_ids = [ln["id"] for ln in lines
                       if ln["end_sec"] >= start - p["line_window"] and ln["start_sec"] <= end + p["line_window"]]
        events.append(ev)
    return TimelineResult(events=events, teams=teams, players=players)


def banner_subtype(text: str) -> str | None:
    """배너 글자 → 오브젝트 종류. 모르면 None."""
    t = text.upper().replace(" ", "")
    if "ELDER" in t:
        return "elder"
    if "BARON" in t:
        return "baron"
    if "HERALD" in t:
        return "herald"
    if "ATAKHAN" in t:
        return "atakhan"
    if "DRAKE" in t or "DRAGON" in t:
        return next((f"dragon:{v}" for k, v in DRAKES.items() if k in t), "dragon")
    return None


def _value_at(series: list[dict], t: float) -> float | None:
    """t 에 보이던 값 (그 시각 run, 없으면 그 앞 마지막 run)."""
    val = None
    for r in series:
        if r["start_sec"] <= t:
            val = float(r["text"])
        else:
            break
    return val


def banner_events(runs: list[dict], p: dict) -> list[tuple]:
    """배너 run → (objective, subtype, 팀, 0, 1, start, end, evidence). 용 팀은 용 아이콘 픽셀 비율이 늘어난 진영."""
    banners = sorted((r for r in runs if r["tag"] == "banner" and banner_subtype(r["text"])), key=lambda r: r["start_sec"])
    icons = {s: sorted((r for r in runs if r["tag"] == "dragon_icons" and r["slot"] == s), key=lambda r: r["start_sec"])
             for s in ("blue", "red")}
    out, last = [], {}
    for r in banners:
        sub = banner_subtype(r["text"])
        kind = sub.split(":")[0]
        if kind in last and r["start_sec"] - last[kind] <= p["banner_merge_sec"]:
            last[kind] = r["start_sec"]
            continue
        last[kind] = r["start_sec"]
        t = r["start_sec"]
        team = None
        if kind in ("dragon", "elder"):
            delta = {s: (_value_at(icons[s], t + p["icon_after_sec"]) or 0) - (_value_at(icons[s], t - p["icon_before_sec"]) or 0)
                     for s in icons if icons[s]}
            if delta and max(delta.values()) >= 0.05:
                team = max(delta, key=delta.get)
        out.append(("objective", sub, team, 0, 1, t - p["icon_before_sec"], t, [f"banner {r['text']!r} @{t}"]))
    return out


def counter_steps(series: list[dict], min_samples: int, max_jump: int) -> list[tuple]:
    """한 진영 카운터 구간 목록 → 값이 오른 시점 목록.

    Args:
        series: 같은 tag · slot 의 roi_ocr 구간 (시간순 아니어도 됨).
        min_samples: 이만큼 이어진 구간만 인정.
        max_jump: 한 번에 이만큼 넘게 오르면 버림.

    Returns:
        [(이전 값, 새 값, start_sec, end_sec, evidence)]. start_sec 는 이전 값이 마지막으로 읽힌 때,
        end_sec 는 새 값이 처음 읽힌 때.
    """
    steps = []
    cur = None       # 지금 인정된 값
    last_seen = None  # 그 값이 마지막으로 읽힌 시각
    for r in sorted(series, key=lambda r: r["start_sec"]):
        if r["samples"] < min_samples or not r["text"].isdigit():
            continue
        v = int(r["text"])
        if cur is None:
            cur, last_seen = v, r["end_sec"]
            continue
        if v < cur or v - cur > max_jump:
            continue  # 줄었거나 너무 많이 오름 → 오독
        if v > cur:
            start = last_seen if last_seen is not None else r["start_sec"]
            steps.append((cur, v, round(start, 2), round(r["start_sec"], 2),
                          [f"{r['tag']}/{r['slot']} {cur}→{v} @{r['start_sec']:.1f}"]))
            cur = v
        last_seen = r["end_sec"]
    return steps


def _kda_series(runs: list[dict], min_samples: int) -> dict[str, list[tuple[float, tuple[int, int, int]]]]:
    """{진영_포지션: [(시작 초, (K, D, A))]} 형식이 맞고 충분히 이어진 KDA 구간만."""
    out: dict[str, list] = {}
    for r in runs:
        m = KDA_RE.match(r["text"]) if r["tag"] == "kda" else None
        if m and r["samples"] >= min_samples:
            out.setdefault(r["slot"], []).append((r["start_sec"], tuple(int(x) for x in m.groups())))
    for v in out.values():
        v.sort()
    return out


def _kill_players(kda: dict, names: dict, team: str, start: float, end: float, window: float) -> dict[str, list[str]]:
    """킬 시점 앞뒤로 KDA 가 바뀐 선수를 killer · victims · assists 로 나눈다."""
    found: dict[str, list[str]] = {}
    for slot, series in kda.items():
        side = slot.split("_")[0]
        for (t0, a), (t1, b) in zip(series, series[1:]):
            if not (start - window <= t1 <= end + window) or any(y < x for x, y in zip(a, b)):
                continue
            name = names.get(slot, slot)
            if side == team and b[0] > a[0]:
                found.setdefault("killer", []).append(name)
            if side == OPPONENT.get(team) and b[1] > a[1]:
                found.setdefault("victims", []).append(name)
            if side == team and b[2] > a[2]:
                found.setdefault("assists", []).append(name)
    return found


def _game_time_at(runs: list[dict], t: float, min_samples: int) -> str | None:
    """t 초에 화면에 떠 있던 경기 시계. 없으면 t 이후 가장 가까운 값."""
    clock = sorted((r for r in runs if r["tag"] == "game_time"), key=lambda r: r["start_sec"])
    for r in clock:
        if r["start_sec"] <= t < r["end_sec"]:
            return r["text"]
    later = [r for r in clock if r["start_sec"] >= t and r["start_sec"] - t <= 5]
    return later[0]["text"] if later else None


def _load_lines(work_dir: Path) -> list[dict]:
    """asr 대사. 아직 없으면 빈 목록."""
    try:
        return json.loads(stage_path(work_dir, "asr").read_text(encoding="utf-8"))["result"]["lines"]
    except FileNotFoundError:
        return []
