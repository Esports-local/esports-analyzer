"""roi_ocr 구간(runs)을 대사 시점의 화면 정보로 정리한다 (translate 조건 B 입력).

esports-api 의 matches/screen.py 와 같은 규칙: 1장짜리 구간은 잡음으로 빼고,
팀·선수 이름처럼 경기 내내 안 바뀌는 값은 화면에 가장 오래 떠 있던 값을 쓴다.
"""

STATIC_TAGS = ("team", "player_name")
SNAPSHOT_TAGS = ("game_time", "team", "team_kills", "player_name", "kda")


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


def snapshot(runs: list[dict], t: float, statics: dict, min_samples: int = 2) -> dict:
    """t 초 화면 정보 {game_time, team, team_kills, player_name, kda}. 없는 값은 빠진다."""
    snap: dict[str, dict[str, str]] = {}
    for r in runs:
        if r["tag"] in SNAPSHOT_TAGS and r["tag"] not in STATIC_TAGS and r["samples"] >= min_samples \
                and r["start_sec"] <= t < r["end_sec"]:
            snap.setdefault(r["tag"], {})[r["slot"] or "value"] = r["text"]
    for tag, values in statics.items():
        snap[tag] = dict(values)
    out = {k: snap[k] for k in SNAPSHOT_TAGS if k in snap}
    if "game_time" in out:
        out["game_time"] = out["game_time"].get("value")
    return out
