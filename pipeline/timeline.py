"""timeline 단계: 화면 숫자(팀 킬 · 타워 · 오브젝트 수)가 오른 시점을 경기 이벤트로 뽑는다.

LLM 을 쓰지 않는다. roi_ocr 결과만으로 코드가 만들기 때문에 같은 입력이면 항상 같은 결과다.
숏폼은 이 이벤트를 기준점으로 구간을 잡는다.

입력: work/roi_ocr.json (필수), work/asr.json (있으면 앞뒤 대사를 붙임)

잡음 정리 (roi_ocr 구간은 1장짜리 오독이 섞여 있다)
- min_samples 장 미만으로 이어진 구간은 버린다
- 팀 킬 · 타워 · 오브젝트 수는 줄지 않는다. 앞 값보다 작은 값은 오독으로 버린다
- 한 번에 max_jump 넘게 오르면 오독으로 보고 버린다 (같은 값이 다시 읽히면 그때 인정)

이벤트 시각
- 화면이 계속 보였으면 새 값이 처음 읽힌 때가 이벤트 시각이다
- 사이에 읽히지 않은 구간(리플레이, 캠 화면)이 있으면 그 사이 어딘가에서 일어났으므로
  start_sec(앞 값이 마지막으로 읽힌 때) ~ end_sec(새 값이 처음 읽힌 때)로 남긴다

선수 (킬만): 이벤트 앞뒤 kda_window 초 안에 KDA 의 K 가 오른 같은 진영 선수 = killer,
D 가 오른 상대 진영 선수 = victims, A 가 오른 같은 진영 선수 = assists. KDA 를 못 읽었으면 비워 둔다.
"""

import json
import logging
import re
from pathlib import Path

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
}

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
    runs = json.loads(stage_path(work_dir, "roi_ocr").read_text(encoding="utf-8"))["result"]["runs"]
    lines = _load_lines(work_dir)
    result = build_timeline(runs, lines, p)
    log.info("[timeline] 이벤트 %d개 (%s)", len(result.events),
             ", ".join(f"{t} {sum(e.type == t for e in result.events)}" for t in COUNTERS.values()))
    return save_stage(work_dir, "timeline", result, p)


def build_timeline(runs: list[dict], lines: list[dict], p: dict) -> TimelineResult:
    """roi_ocr 구간과 대사로 이벤트 목록을 만든다 (파일 입출력 없음, 테스트용)."""
    statics = static_values(runs)
    teams = statics.get("team", {})
    players = statics.get("player_name", {})
    raw = []
    for tag, etype in COUNTERS.items():
        for slot in ("blue", "red"):
            series = [r for r in runs if r["tag"] == tag and r["slot"] == slot]
            raw += [(etype, slot, *step) for step in counter_steps(series, p["min_samples"], p["max_jump"])]
    raw.sort(key=lambda x: (x[4], x[1]))  # end_sec(새 값이 읽힌 때), 진영

    kda = _kda_series(runs, p["min_samples"])
    events = []
    for i, (etype, slot, before, after, start, end, evidence) in enumerate(raw, 1):
        ev = TimelineEvent(
            id=i, type=etype, team=slot, team_name=teams.get(slot),
            start_sec=start, end_sec=end, value_before=before, value_after=after,
            game_time=_game_time_at(runs, end, p["min_samples"]), evidence=evidence,
        )
        if etype == "kill":
            ev.players = _kill_players(kda, players, slot, start, end, p["kda_window"])
        ev.line_ids = [ln["id"] for ln in lines
                       if ln["end_sec"] >= start - p["line_window"] and ln["start_sec"] <= end + p["line_window"]]
        events.append(ev)
    return TimelineResult(events=events, teams=teams, players=players)


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
