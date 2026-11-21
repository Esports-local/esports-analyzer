"""context 단계 (바디): 영상 기준 경기 데이터. 숏폼·자막 교정이 쓰는 사실만 담는다 (하이라이트 판단은 shorts 쪽).

우리 분석 결과만으로 만든다 (op.gg 세트 기록 opgg_timeline.json 은 쓰지 않는다, 마지막 검증의 정답지).
LLM 없음. 같은 입력이면 항상 같은 결과.

입력 (work/): header(선수·챔피언), roi_ocr(경기 시간·골드, 필수), timeline(이벤트), segments(구간 종류·전환),
asr(대사), audio.wav(소리 크기). header · segments · asr · wav 는 없으면 그 부분만 비운다.

결과 (work/context.json 의 result)
    game:     {start_sec, end_sec, length_sec, winner, checks}   경기 00:00 ~ 끝 (영상 초)
    players:  {슬롯: {team, name, champion, champion_en}}         슬롯 = 진영_포지션 (header)
    clock:    [{start_sec, end_sec, game_start_sec}]              경기 시간이 흐르는 구간. 영상 t → 경기 t - start_sec + game_start_sec
    segments: [{id, start_sec, end_sec, kind, transition_in, game_time}]  컷 사이 구간과 화면 종류
    events:   [{id, t, start_sec, end_sec, game_time, type, team, killer, victims, assists, tags, segment, line_ids, source}]
    signals:  {step_sec, start_sec, loudness[], gold_diff[], cut_rate[], kind[]}   초 단위

clock: 화면 경기 시간 구간마다 (영상 시각 - 경기 시각) 차이를 구해, 차이가 같은 구간끼리 묶는다.
리플레이·쉬는 시간에는 경기 시간이 안 읽히거나 멈춰서 차이가 바뀌므로 자연히 다른 묶음이 된다.

events 태그: first_blood(첫 킬) · multikill(같은 선수가 multikill_sec 안에 2킬 이상, count) ·
teamfight(teamfight_sec 간격으로 이어진 킬이 teamfight_kills 개 이상인 묶음, fight id).
"""

import json
import logging
import re
import statistics
import wave
from pathlib import Path

import numpy as np

from pipeline.io import save_stage, stage_path
from pipeline.schema import ContextResult

log = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "clock_tolerance_sec": 1.5,  # 영상-경기 차이가 이만큼 안이면 같은 clock 구간
    "clock_min_sec": 3.0,        # 이보다 짧은 clock 구간은 버린다 (오독)
    "clock_gap_sec": 3.0,        # 경기 시간이 이보다 오래 안 읽히면 (리플레이·캠) clock 구간을 끊는다
    "multikill_sec": 10.0,
    "teamfight_sec": 15.0,
    "teamfight_kills": 3,
    "cut_window_sec": 5.0,       # cut_rate: 앞뒤 이 초 안 전환 수
    "step_sec": 1.0,
}
CLOCK_RE = re.compile(r"^(\d{1,2}):(\d{2})$")
GOLD_RE = re.compile(r"^(\d{1,3}(?:\.\d)?)K$")


def run(video: Path, work_dir: Path, params: dict | None = None) -> Path:
    """work/context.json 을 만든다. 영상은 읽지 않는다 (소리 크기는 work/audio.wav)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    load = lambda s: _result(work_dir, s)
    roi = load("roi_ocr")
    if roi is None:
        raise FileNotFoundError("context 는 work/roi_ocr.json 이 필요합니다 (roi_ocr 단계 먼저)")
    pre = load("preprocess") or {}
    result = build_context(
        header=load("header"), runs=roi["runs"], timeline=load("timeline"), segments=load("segments"),
        lines=(load("asr") or {}).get("lines", []),
        duration=pre.get("duration_sec") or max((r["end_sec"] for r in roi["runs"]), default=0.0),
        loudness=loudness(work_dir / "work" / "audio.wav", p["step_sec"]), p=p,
        start=pre.get("range_start_sec") or 0.0)
    log.info("context: clock %d구간, 구간 %d, 이벤트 %d, 경기 %s", len(result.clock), len(result.segments),
             len(result.events), result.game)
    return save_stage(work_dir, "context", result, p)


def _result(work_dir: Path, stage: str) -> dict | None:
    path = stage_path(work_dir, stage)
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8")).get("result")
    return None if (result or {}).get("error") else result


def build_context(header: dict | None, runs: list[dict], timeline: dict | None, segments: dict | None,
                  lines: list[dict], duration: float, loudness: list[float] | None, p: dict,
                  start: float = 0.0) -> ContextResult:
    players = build_players(header)
    clock = build_clock(runs, p["clock_tolerance_sec"], p["clock_min_sec"], p["clock_gap_sec"])
    segs = build_segments(segments)
    events = build_events(timeline, players, segs, p)
    return ContextResult(
        game=build_game(header, segments, clock), players=players, clock=clock, segments=segs, events=events,
        signals=build_signals(duration, runs, segments, segs, loudness, p, start))


# ---------------------------------------------------------------- 경기 · 선수 · 시계

def build_players(header: dict | None) -> dict[str, dict]:
    """{슬롯: {team, name, champion, champion_en}} (header). header 가 없으면 빈 칸."""
    out = {}
    for pl in (header or {}).get("players", []):
        champ = pl.get("champion") or {}
        out[f"{pl['side']}_{pl['role']}"] = {"team": pl.get("team"), "name": pl.get("name"),
                                             "champion": champ.get("ko"), "champion_en": champ.get("name")}
    return out


def clock_sec(text: str | None) -> int | None:
    m = CLOCK_RE.match(text or "")
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def build_clock(runs: list[dict], tolerance: float, min_sec: float, gap: float = 3.0) -> list[dict]:
    """화면 경기 시간 구간들 → [{start_sec, end_sec, game_start_sec}].

    경기 시간 run 마다 차이 = 그 값이 처음 보인 영상 시각 - 경기 초. 차이가 tolerance 안이고 사이가 gap 초 안인
    run 들을 시간순으로 묶고, 묶음의 차이는 중앙값. 짧은 묶음(min_sec 미만)은 오독으로 버리고,
    그 앞뒤 묶음의 차이가 같고 사이가 gap 안이면 다시 하나로 합친다. 경기 시간이 안 보인 리플레이·캠 동안은 clock 밖이 된다.
    경기 시간은 거꾸로 가지 않으므로, 앞 구간 끝 경기 시각보다 tolerance 넘게 이른 데서 시작하는 묶음은 버린다 (오독·지난 장면).
    """
    marks = sorted((r["start_sec"], r["end_sec"], g) for r in runs
                   if r["tag"] == "game_time" and (g := clock_sec(r["text"])) is not None)
    groups: list[list[tuple]] = []
    for m in marks:
        if groups and m[0] - groups[-1][-1][1] <= gap \
                and abs((m[0] - m[2]) - statistics.median(x[0] - x[2] for x in groups[-1])) <= tolerance:
            groups[-1].append(m)
        else:
            groups.append([m])
    offset = lambda g: statistics.median(x[0] - x[2] for x in g)
    kept: list[list[tuple]] = []
    for g in groups:
        if g[-1][1] - g[0][0] < min_sec:
            continue
        if kept and g[0][0] - kept[-1][-1][1] <= gap and abs(offset(kept[-1]) - offset(g)) <= tolerance:
            kept[-1] += g
        else:
            kept.append(g)
    out, game_end = [], float("-inf")
    for g in kept:
        start, end, off = g[0][0], g[-1][1], offset(g)
        if start - off < game_end - tolerance:
            continue
        out.append({"start_sec": round(start, 3), "end_sec": round(end, 3), "game_start_sec": round(start - off, 3)})
        game_end = end - off
    return out


def video_to_game(clock: list[dict], t: float) -> float | None:
    """영상 t 초 → 경기 초. 경기 시간이 흐르지 않는 때(리플레이 등)면 None."""
    for c in clock:
        if c["start_sec"] <= t < c["end_sec"]:
            return t - c["start_sec"] + c["game_start_sec"]
    return None


def game_to_video(clock: list[dict], g: float) -> float | None:
    """경기 g 초 → 영상 초 (그 경기 시각이 화면에 나온 구간). 없으면 None."""
    for c in clock:
        t = g - c["game_start_sec"] + c["start_sec"]
        if c["start_sec"] <= t < c["end_sec"]:
            return t
    return None


def build_game(header: dict | None, segments: dict | None, clock: list[dict]) -> dict:
    """경기 구간. segments 의 game(화면 판별)이 있으면 그것, 없으면 clock 으로."""
    res = (header or {}).get("result") or {}
    game = dict((segments or {}).get("game") or {})
    if not game and clock:
        game = {"start_sec": round(clock[0]["start_sec"] - clock[0]["game_start_sec"], 3), "end_sec": clock[-1]["end_sec"]}
    if not game:
        return {}
    return {"start_sec": game.get("start_sec"), "end_sec": game.get("end_sec"),
            "length_sec": res.get("length_sec") or game.get("header_length_sec"), "winner": res.get("winner"),
            "checks": game.get("checks")}


# ---------------------------------------------------------------- 구간 · 이벤트

def build_segments(segments: dict | None) -> list[dict]:
    """segments 결과 → [{id, start_sec, end_sec, kind, transition_in, game_time}]. transition_in 은 들어온 전환 종류."""
    if not segments:
        return []
    kinds = {t["id"]: t["kind"] for t in segments.get("transitions", [])}
    out = []
    for s in segments.get("segments", []):
        gt = [s["game_start"], s["game_end"]] if s.get("game_start") else None
        out.append({"id": s["id"], "start_sec": s["start_sec"], "end_sec": s["end_sec"], "kind": s.get("kind"),
                    "transition_in": kinds.get(s.get("transition_in")), "game_time": gt})
    return out


def kind_at(segs: list[dict], t: float) -> str | None:
    for s in segs:
        if s["start_sec"] <= t < s["end_sec"]:
            return s["kind"]
    return None


def build_events(timeline: dict | None, players: dict, segs: list[dict], p: dict) -> list[dict]:
    """timeline 이벤트 → 슬롯·태그·화면 종류를 붙인 이벤트. 선수는 화면 이름 → 슬롯 (timeline.players)."""
    if not timeline:
        return []
    slot_of = {name: slot for slot, name in (timeline.get("players") or {}).items()}
    to_slots = lambda names: [slot_of.get(n, n) for n in names or []]
    out = []
    for e in timeline.get("events", []):
        pl = e.get("players") or {}
        killer = to_slots(pl.get("killer"))
        out.append({"id": e["id"], "t": e["end_sec"], "start_sec": e["start_sec"], "end_sec": e["end_sec"],
                    "game_time": e.get("game_time"), "type": e["type"], "team": e["team"],
                    "killer": killer[0] if killer else None, "victims": to_slots(pl.get("victims")),
                    "assists": to_slots(pl.get("assists")), "tags": [], "segment": kind_at(segs, e["end_sec"]),
                    "line_ids": e.get("line_ids", []), "source": e.get("evidence", [])})
    tag_kills(out, p)
    return out


def tag_kills(events: list[dict], p: dict) -> None:
    """킬 이벤트에 first_blood · multikill · teamfight 태그를 붙인다 (제자리)."""
    kills = [e for e in events if e["type"] == "kill"]
    if kills:
        kills[0]["tags"].append("first_blood")
    for i, e in enumerate(kills):  # 같은 선수 연속 킬
        if not e["killer"]:
            continue
        chain = [k for k in kills[: i + 1] if k["killer"] == e["killer"] and e["t"] - k["t"] <= p["multikill_sec"]]
        if len(chain) >= 2:
            e["tags"] = [t for t in e["tags"] if not t.startswith("multikill")] + [f"multikill:{len(chain)}"]
    fight, group = 0, []
    for e in kills + [None]:  # 이어진 킬 묶음
        if e is not None and group and e["t"] - group[-1]["t"] <= p["teamfight_sec"]:
            group.append(e)
            continue
        if len(group) >= p["teamfight_kills"]:
            fight += 1
            for k in group:
                k["tags"].append(f"teamfight:{fight}")
        group = [e] if e is not None else []


# ---------------------------------------------------------------- 초 단위 신호

def loudness(wav: Path, step: float) -> list[float] | None:
    """audio.wav(16bit) → step 초마다 소리 크기 0~1 (RMS dB 를 하위 5% ~ 상위 1% 로 맞춤). 파일이 없으면 None."""
    if not wav.exists():
        return None
    with wave.open(str(wav)) as w:
        rate, ch = w.getframerate(), w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    n = int(rate * step)
    if n <= 0 or len(data) < n:
        return []
    frames = data[: len(data) // n * n].reshape(-1, n)
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-6)
    lo, hi = np.percentile(db, 5), np.percentile(db, 99)
    return [round(float(x), 3) for x in np.clip((db - lo) / max(hi - lo, 1e-6), 0, 1)]


def gold_value(text: str) -> float | None:
    m = GOLD_RE.match(text or "")
    return float(m.group(1)) * 1000 if m else None


def build_signals(duration: float, runs: list[dict], segments: dict | None, segs: list[dict],
                  loud: list[float] | None, p: dict, start: float = 0.0) -> dict:
    """step 초 간격 신호. 분석한 구간(start ~ duration)만 담는다: 칸 i = 영상 start + i*step 초.

    gold_diff 는 블루 - 레드 (마지막으로 읽힌 값을 이어 씀, 읽기 전에는 None). loudness 는 audio.wav 가
    이미 start 부터 뽑혀 있으므로 그대로 쓴다.
    """
    step = p["step_sec"]
    n = max(0, int((duration - start) // step) + 1)
    times = [start + i * step for i in range(n)]
    idx = lambda t: int((t - start) // step)

    gold = {"blue": [None] * n, "red": [None] * n}
    for r in runs:
        if r["tag"] == "gold" and r["slot"] in gold and (v := gold_value(r["text"])) is not None:
            for i in range(max(0, idx(r["start_sec"])), min(n, idx(r["end_sec"]) + 1)):
                gold[r["slot"]][i] = v
    diff, last = [], {"blue": None, "red": None}
    for i in range(n):
        for side in last:
            last[side] = gold[side][i] if gold[side][i] is not None else last[side]
        diff.append(None if None in last.values() else last["blue"] - last["red"])

    cuts = [t["start_sec"] for t in (segments or {}).get("transitions", [])]
    w = p["cut_window_sec"]
    cut_rate = [sum(1 for c in cuts if abs(c - t) <= w) for t in times] if cuts else []

    return {"step_sec": step, "start_sec": start, "loudness": (loud or [])[:n], "gold_diff": diff,
            "cut_rate": cut_rate, "kind": [kind_at(segs, t) for t in times] if segs else []}
